#!/usr/bin/env/python3
"""Recipe for training a sequence-to-sequence dialogue model.
The system employs a Transformer.

To run this recipe, do the following:
> python train.py hparams/Transformers.yaml

The neural network is trained with the negative-log likelihood objective and
characters are used as basic tokens for both context and reply.
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
#import sacrebleu
import math

logger = logging.getLogger(__name__)


# Brain class for speech recognition training
class DialogueTransformer(sb.Brain):
    """Class that manages the training loop."""

    def compute_forward(self, batch, stage):
        """Computations from the input to the output representing 
        the generated dialogue reply conditioned on the input.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions, hyps : torch.tensor
            predictions: Log-probabilities predicted by the decoder.
            hyps: At validation/test time, it returns the predicted tokens as well.
        """
        # first move the batch to the appropriate device.
        batch = batch.to(self.device)
        
        # unpack input
        enc_history, inp_lens = batch.history_encoded_chars        
        enc_reply_bos,  out_lens = batch.reply_encoded_chars_bos  
        
        # input embeddings
        enc_emb = self.modules.encoder_emb(enc_history)            
        
        # positional embeddings
        pos_emb_enc = self.modules.pos_emb_enc(enc_emb)
        
        # sum up embeddings
        enc_emb = enc_emb + pos_emb_enc

        # decoding embeddings
        dec_emb = self.modules.decoder_emb(enc_reply_bos)
        pos_emb_dec = self.modules.pos_emb_dec(dec_emb)
        dec_emb = dec_emb + pos_emb_dec 
        
        # get target mask (to avoid looking ahead)
        tgt_mask = self.hparams.lookahead_mask(enc_reply_bos) 
        
        # get the source mask (all zeros is fine in this case to allow the
        # network to embed both past and future history)
        src_mask = torch.zeros(enc_history.shape[1], enc_history.shape[1])
        
        # padding masks for source and targets (use padding_mask)
        src_key_padding_mask = self.hparams.padding_mask(padded_input=enc_history, pad_idx=self.hparams.blank_index)  
        tgt_key_padding_mask = self.hparams.padding_mask(padded_input=enc_reply_bos, pad_idx=self.hparams.blank_index)  
        
        # run the Seq2Seq Transformer
        decoder_outputs = self.modules.Seq2SeqTransformer(
            src = enc_emb, 
            tgt = dec_emb, 
            src_mask = src_mask, 
            tgt_mask = tgt_mask, 
            src_key_padding_mask = src_key_padding_mask, 
            tgt_key_padding_mask = tgt_key_padding_mask
        ).to(self.device)
        
        # compute logits
        logits = self.modules.seq_lin(decoder_outputs)
        
        # apply log softmax
        predictions = self.hparams.log_softmax(logits)
           
        #  Generate Hypothesis for validation and test using greedy search
        if stage != sb.Stage.TRAIN:
            
            # Greedy Decoding
            hyps = predictions.argmax(-1)
            
            # getting the first index where the prediciton is eos_index
            stop_indexes = (hyps==self.hparams.eos_index).int()
            stop_indexes = stop_indexes.argmax(dim=1)
            
            # Converting hyps from indexes to chars
            hyp_lst = []
            for hyp, stop_ind in zip(hyps, stop_indexes):
                # in some cases the eos in not observed (e.g, for the last sentence
                # in the batch)
                if stop_ind == 0:
                    stop_ind = -1
                # Stopping when eos is observed
                hyp = hyp[0:stop_ind]
                # From index to character
                hyp_lst.append(self.label_encoder.decode_ndim(hyp))
            return predictions, hyp_lst
        
        return predictions, None

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
        # unpack labels
        ids = batch.id
        enc_reply_eos, reply_lens = batch.reply_encoded_chars_eos

        # unpack predictions
        predictions, hyp_lst = predictions

        
        # compute loss
        loss = self.hparams.seq_cost(predictions, 
                                     enc_reply_eos, 
                                     length=reply_lens)

        if stage != sb.Stage.TRAIN:
          
          for id, label, hyp in zip(batch.id, batch.reply_chars, hyp_lst):
              # get the target and predicted words in the necessary format
              target_words = [''.join(label).split(' ')]
              predicted_words = [''.join(hyp).split(' ')]

              self.wer_metric.append(id, predicted_words, target_words)
              self.cer_metric.append(id, predicted_words, target_words)
              #self.bleu_metric.append(id, predicted_words, target_words)
              
        
        return loss

    def on_stage_start(self, stage, epoch): 
        """Gets called at the beginning of each epoch""" 
        
        if stage != sb.Stage.TRAIN: 
            self.cer_metric = self.hparams.cer_computer() 
            self.wer_metric = self.hparams.error_rate_computer() 
            self.bleu_metric = self.hparams.bleu_computer()
            self.acc_metric = self.hparams.acc_computer()

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

        # compute stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            stage_stats["SER"] = self.wer_metric.summarize("SER")

            # TODO scarebleu not working in Gradient enviroment
            stage_stats["BLEU"] = self.bleu_metric.summarize("BLEU")
            perplexity = math.e**stage_loss
            stage_stats["perplexity"] = perplexity


        # perform end-of-iteration things, like annealing, logging, etc.
        # learning rate annealing and checkpoint
        if stage == sb.Stage.VALID:
                
            # update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # write a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]}, min_keys=["loss"],
            )
        # write statistics about test data to stdout and to the logfile   
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

            # TODO - why BLEU gives an error: sacrebleu - IndexError: list index out of range
            with open(self.hparams.bleu_file, "w", encoding="utf-8") as w:
                self.bleu_metric.write_stats(w)
            
        
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

    # Define datasets from json data manifest file
    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    
    # The label encoder will assign a different integer to each element
    # in the output vocabulary
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


    # Load or compute the label encoder
    inp_enc_file = os.path.join(hparams["save_folder"], "input_encoder.txt")
    
    # The blank symbol is used to indicate padding
    special_labels = {"blank_label": hparams["blank_index"]}
    input_encoder.add_unk()
    input_encoder.load_or_create(
        path=inp_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="history_chars",
        special_labels=special_labels,
        sequence_input=True,
    )
        
    # Load or compute the label encoder
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
    dialogue_brain = DialogueTransformer(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    
    # make label encoder accessible (needed for computer the character error rate)
    dialogue_brain.label_encoder = label_encoder


    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    dialogue_brain.fit(
        dialogue_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Load best checkpoint for evaluation
    test_stats = dialogue_brain.evaluate(
        test_set=datasets["test"],
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
    
