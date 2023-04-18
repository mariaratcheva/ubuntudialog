
"""
Seq-to-seq generative dialogue model - baseline.
The model employs an encoder, a decoder, and a Bahdanau's cross attention 
mechanism to generate the next utterance of a dialogue. 

To run this recipe, do the following:
> python train_base_GRU.py harams_base_GRU.yaml

The neural network is trained with the negative-log likelihood objective and
characters are used as basic tokens for both dialogue history and reply.
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import math

logger = logging.getLogger(__name__)


# Brain class for dialogue model training
class Dialogue(sb.Brain):
    """Class that manages the training loop."""

    def compute_forward(self, batch, stage):
        """Computations from the input representing a single dialogue turn
        to the output representing the generated reply.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions, inp_lens, hyps : torch.tensor
            predictions : Log-probabilities predicted by the decoder.
            inp_lens: The actual length of each input sequence
            hyps: At validation/test time, we use beamsearch to find the most likely 
            output sequence based on the decoder output probabilities.
        """
        # move the batch to the appropriate device.
        batch = batch.to(self.device)
   
        # unpack history tokens
        enc_history, inp_lens = batch.history_encoded_chars 
        
        # get the input embeddings
        enc_emb = self.modules.encoder_emb(enc_history) 

        # run the encoder
        encoder_output, _ = self.modules.encoder(enc_emb)

        # unpack reply tokens - the labels
        enc_reply_bos, reply_lens = batch.reply_encoded_chars_bos 

        # get decoding embeddings
        dec_emb = self.modules.decoder_emb(enc_reply_bos)

        # run the decoder
        decoder_outputs, _ = self.modules.decoder(dec_emb, encoder_output, inp_lens)
        
        # output layer for seq2seq log-probabilities
        # compute logits i.e., apply final linear transformation
        logits =  self.modules.seq_lin(decoder_outputs)

        # compute log posteriors
        predictions = self.hparams.log_softmax(logits)

        # generate Hypothesis for validation and test using greedy and beam search
        hyps = None
        if stage != sb.Stage.TRAIN:
          if stage == sb.Stage.VALID:
            hyps, scores = self.hparams.greedy_searcher(encoder_output, inp_lens)
          elif stage == sb.Stage.TEST:
            hyps, scores = self.hparams.beam_searcher(encoder_output, inp_lens)
           
        return predictions, inp_lens, hyps
        

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        
        Arguments
        ---------
        predictions : torch.tTensor
            The output tensor from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        current_epoch = self.hparams.epoch_counter.current
        
        predictions, inp_lens, predicted_tokens = predictions

        # unpack labels and labels with eos    
        ids = batch.id
        reply_encoded_chars_eos, reply_encoded_chars_eos_lens = batch.reply_encoded_chars_eos
        reply_encoded_chars, reply_encoded_chars_lens = batch.reply_encoded_chars

        # compute loss
        loss = self.hparams.seq_cost(predictions, 
                                     reply_encoded_chars_eos, 
                                     length=reply_encoded_chars_eos_lens)
        
        # the tokenizer is label_encoder
        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = [
                "".join(self.tokenizer.decode_ndim(utt_seq)).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.reply]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        # avoid error: Loss is NaN
        eps = 1e-6
        if loss.isnan(): 
          loss=eps
          print("**LOSS IS NaN**")  

        return loss
        

    def on_stage_start(self, stage, epoch): 
        """Gets called at the beginning of each epoch""" 
        if stage != sb.Stage.TRAIN: 
            self.cer_metric = self.hparams.cer_computer() 
            self.wer_metric = self.hparams.error_rate_computer() 
            self.acc_metric = self.hparams.acc_computer()  
            self.bleu_metric = self.hparams.bleu_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            stage_stats["SER"] = self.wer_metric.summarize("SER")
            perplexity = math.e**stage_loss
            stage_stats["perplexity"] = perplexity

        # perform end-of-iteration things, like annealing, logging, etc.
        # learning rate annealing and checkpoint are based on WER
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)


def dataio_prepare(hparams):
    # Define text processing pipeline. We start from the raw text and then
    # split it into characters. The tokens with BOS are used for feeding
    # the decoder during training (right shifr), the tokens with EOS 
    # are used for computing the cost function.
    @sb.utils.data_pipeline.takes("reply")
    @sb.utils.data_pipeline.provides(
        "reply",
        "reply_chars",
        "reply_encoded_chars_lst",
        "reply_encoded_chars",
        "reply_encoded_chars_eos",
        "reply_encoded_chars_bos",
        )
    def reply_text_pipeline(reply):
        yield reply
        reply_chars = list(reply)
        yield reply_chars
        reply_encoded_chars_lst = label_encoder.encode_sequence(reply_chars)
        yield reply_encoded_chars_lst 
        reply_encoded_chars = torch.LongTensor(reply_encoded_chars_lst)
        yield reply_encoded_chars
        reply_encoded_chars_eos = torch.LongTensor(label_encoder.append_eos_index(reply_encoded_chars_lst))
        yield reply_encoded_chars_eos                                              
        reply_encoded_chars_bos = torch.LongTensor(label_encoder.prepend_bos_index(reply_encoded_chars_lst))
        yield reply_encoded_chars_bos  

    @sb.utils.data_pipeline.takes("history")
    @sb.utils.data_pipeline.provides("history", "history_chars", "history_encoded_chars")
    def history_text_pipeline(history):
        yield history
        history_chars = list(history)
        yield history_chars
        history_encoded_chars = torch.LongTensor(input_encoder.encode_sequence(history_chars))
        yield history_encoded_chars

    # define datasets 
    datasets = {}
    data_info = {
        "train": hparams["train_json"],
        "valid": hparams["valid_json"],
        "test": hparams["test_json"],
    }
    
    input_encoder = sb.dataio.encoder.CTCTextEncoder()
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            dynamic_items=[reply_text_pipeline, history_text_pipeline],
            output_keys=[
                "id",
                "reply",
                "reply_chars",
                "reply_encoded_chars",
                "reply_encoded_chars_eos",
                "reply_encoded_chars_bos",
                "history",
                "history_chars",
                "history_encoded_chars",
            ],
        )
        # sort by length
        if hparams["sorting"] == "ascending":
            # we sort training data to speed up training and get better results.
            datasets[dataset] = datasets[dataset].filtered_sorted(
                sort_key="length",
                key_max_value={"length": hparams["avoid_if_longer_than"]},
            )
            # when sorting do not shuffle in dataloader ! otherwise is pointless
            hparams[f"{dataset}_dataloader_opts"]["shuffle"] = False

        elif hparams["sorting"] == "descending":
            datasets[dataset] = datasets[dataset].filtered_sorted(
                sort_key="length",
                reverse=True,
                key_max_value={"length": hparams["avoid_if_longer_than"]},
            )
            # when sorting do not shuffle in dataloader ! otherwise is pointless
            hparams[f"{dataset}_dataloader_opts"]["shuffle"] = False

        elif hparams["sorting"] == "random":
              hparams[f"{dataset}_dataloader_opts"]["shuffle"] = True

        else:
            raise NotImplementedError(
                "sorting must be random, ascending or descending"
            )


    # load or compute the label encoder for inputs
    inp_enc_file = os.path.join(hparams["save_folder"], "input_encoder.txt")
    
    # blank symbol used to indicate padding
    special_labels = {"blank_label": hparams["blank_index"]}
    input_encoder.add_unk()
    input_encoder.load_or_create(
        path=inp_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="history_chars",
        special_labels=special_labels,
        sequence_input=True,
    )
        
    # load or compute the label encoder for output labels
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "blank_label": hparams["blank_index"],
        "bos_label": hparams["bos_index"],
        "eos_label": hparams["eos_index"],
    }
    label_encoder.add_unk()
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="reply_chars",
        special_labels=special_labels,
        sequence_input=True,
    )
    
    return datasets, label_encoder

if __name__ == "__main__":

    # read command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # create the datasets for training, valid, and test
    datasets, label_encoder = dataio_prepare(hparams)

    # trainer initialization
    dialogue_brain = Dialogue(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    
    # making label encoder accessible (needed for computing the character/word error rate)
    dialogue_brain.tokenizer = label_encoder

    # training
    dialogue_brain.fit(
          dialogue_brain.hparams.epoch_counter,
          datasets["train"],
          datasets["valid"],
          train_loader_kwargs=hparams["train_dataloader_opts"],
          valid_loader_kwargs=hparams["valid_dataloader_opts"],
      )

    # save the WER file
    dialogue_brain.hparams.wer_file = hparams["output_folder"] + "/wer.txt"

    # load best checkpoint for evaluation
    test_stats = dialogue_brain.evaluate(
        test_set=datasets["test"],
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
