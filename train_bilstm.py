
import os
import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
import math

logger = logging.getLogger(__name__)


# Define training procedure
class DialogueBase(sb.Brain):
    
    
    def compute_forward(self, batch, stage):
        """Computations from input history to semantic outputs"""
        
        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)
        
        # unpack history tokens
        history_tokens, history_tokens_lens = batch.history_tokens
        
        # get the input embeddings 
        embedded_history = self.modules.input_emb(history_tokens)

        # run the encoder
        encoder_output = self.modules.encoder(embedded_history)

        # unpack reply tokens - the labels
        reply_tokens_bos, reply_tokens_bos_lens = batch.reply_tokens_bos

        # get decoding embeddings
        dec_emb = self.modules.output_emb(reply_tokens_bos)

        # run the decoder
        decoder_outputs, _ = self.modules.dec(dec_emb, encoder_output, history_tokens_lens)

        # Output layer for seq2seq log-probabilities
        # Compute logits (i.e., apply final linear transformation)
        logits = self.modules.seq_lin(decoder_outputs)

        # Compute log posteriors - predictions
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        if (stage == sb.Stage.TRAIN):
            return p_seq, history_tokens_lens
        else:
            # use beam search to generate the hypothesis - the most likely 
            # output sequence based on the decoder output probabilities
            p_tokens, scores = self.hparams.beam_searcher(
                encoder_output, history_tokens_lens
            )
            return p_seq, history_tokens_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""

        current_epoch = self.hparams.epoch_counter.current
        if stage == sb.Stage.TRAIN:
            p_seq, history_tokens_lens = predictions
        else:
            p_seq, history_tokens_lens, predicted_tokens = predictions
        
        # unpacking labels
        ids = batch.id
        reply_tokens_eos, reply_tokens_eos_lens = batch.reply_tokens_eos
        reply_tokens, reply_tokens_lens = batch.reply_tokens

        # compute loss
        loss = self.hparams.seq_cost(p_seq, reply_tokens_eos, length=reply_tokens_eos_lens)

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = [
                self.tokenizer_reply.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.reply]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)
            perplexity = math.e**stage_loss
            stage_stats["perplexity"] = perplexity

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # compute stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            stage_stats["SER"] = self.wer_metric.summarize("SER")

        # perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # update learning rate
            #old_lr, new_lr = self.hparams.lr_annealing(epoch)
            #sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            
            # update learning rate
            #old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            #sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # save checkpoint
            #self.checkpointer.save_and_keep_only(
            #    meta={"SER": stage_stats["SER"]}, min_keys=["SER"],
            #)

            # save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]}, min_keys=["loss"],
            )
            
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    # 1. Declarations:

    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_json"],
        #replacements = replacements_all
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_json"],
        #replacements = replacements_all
    )
  
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_json"],
        #replacements = replacements_all
    )
 
    datasets = [train_data, valid_data, test_data]

    # We get the tokenizers as we need them to encode the labels and the input when creating mini-batches
    tokenizer_history = hparams["tokenizer_history"]
    tokenizer_reply = hparams["tokenizer_reply"]

    # 2. Define input pipeline:
    @sb.utils.data_pipeline.takes("history")
    @sb.utils.data_pipeline.provides("history", "history_tokens")
    def history_pipeline(history):
        yield history
        history_tokens_list = tokenizer_history.encode_as_ids(history)
        history_tokens = torch.LongTensor(history_tokens_list)
        yield history_tokens

    sb.dataio.dataset.add_dynamic_item(datasets, history_pipeline)

    # 3. Define output pipeline:
    @sb.utils.data_pipeline.takes("reply")
    @sb.utils.data_pipeline.provides(
        "reply",
        "reply_tokens_list",
        "reply_tokens_bos",
        "reply_tokens_eos",
        "reply_tokens",
    )
    def reply_pipeline(reply):
        yield reply
        reply_tokens_list = tokenizer_reply.encode_as_ids(reply)
        yield reply_tokens_list
        reply_tokens_bos = torch.LongTensor(
            [hparams["bos_index"]] + (reply_tokens_list)
        )
        yield reply_tokens_bos
        reply_tokens_eos = torch.LongTensor(
            reply_tokens_list + [hparams["eos_index"]]
        )
        yield reply_tokens_eos
        reply_tokens = torch.LongTensor(reply_tokens_list)
        yield reply_tokens

    sb.dataio.dataset.add_dynamic_item(datasets, reply_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "history",
            "history_tokens",
            "reply",
            "reply_tokens_bos",
            "reply_tokens_eos",
            "reply_tokens",
        ],
    )

    return (
        train_data,
        valid_data,
        test_data
    )


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hparams with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_set,
        valid_set,
        test_set,
    ) = dataio_prepare(hparams)


    # We download the pretrained ASR from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    dialogue_brain = DialogueBase(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We dynamicaly add the tokenizer to our brain class.
    dialogue_brain.tokenizer_history = hparams["tokenizer_history"]
    dialogue_brain.tokenizer_reply = hparams["tokenizer_reply"]

    # Training
    dialogue_brain.fit(
        dialogue_brain.hparams.epoch_counter,
        train_set,
        valid_set,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    # save the WER file
    dialogue_brain.hparams.wer_file = hparams["output_folder"] + "/wer.txt"

    # Testing
    dialogue_brain.evaluate(
        test_set,
        min_key="loss",
        test_loader_kwargs=hparams["dataloader_opts"],
    )
