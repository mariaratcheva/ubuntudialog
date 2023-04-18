
import matplotlib.pyplot as plt

def visualize_logs(log_file, model=""):

    def get_losses(log_file):
      """This function takes in input a path of a log-file and outputs 
      the log information as lists of float numbers"""

      x_epochs_list =[]
      train_loss_list = []
      valid_loss_list = []
      wer_list=[]
      bleu_list=[]
      perplexity_list = []
      with open(log_file) as f:
        lines=f.readlines()
        result=[]
        for x in lines:
            if x.split(' ')[0] != 'Epoch':
              x_epochs_list.append(int(x.split(' ')[1].split(',')[0]))
              train_loss_list.append(float((x.split(' ')[7])))
              valid_loss_list.append(float(x.split(' ')[11].split(',')[0]))
              wer_list.append(float(x.split(' ')[17].split(',')[0]))
              if len(x.split(' ')) > 20 :
                  bleu_list.append(float(x.split(' ')[20].split(',')[0]))
                  perplexity_list.append(float(x.split(' ')[23]))

      return x_epochs_list, train_loss_list, valid_loss_list, wer_list, bleu_list, perplexity_list


    x_epochs, train_losses, valid_losses, wer, bleu, perplexity = get_losses(log_file)

    fig, axs = plt.subplots(2,2)
    fig.suptitle(model)
    fig.tight_layout(pad=3)
    default_x_ticks = range(1, len(x_epochs)+1)
    axs[0,0].plot(default_x_ticks, train_losses, label='train')
    axs[0,0].plot(default_x_ticks, valid_losses, label='valid')
    axs[0,0].set_xticks(default_x_ticks, minor=False)
    axs[0,0].set_ylabel('Loss')
    axs[0,0].set_xlabel('# Epochs')
    axs[0,0].legend()

    axs[0,1].plot(default_x_ticks, perplexity, label='perplexity')
    axs[0,1].set_xticks(default_x_ticks, minor=False)
    axs[0,1].set_ylabel('perplexity')
    axs[0,1].set_xlabel('# Epochs')
    axs[0,1].legend()

    axs[1,0].plot(default_x_ticks, wer, label='WER')
    axs[1,0].set_xticks(default_x_ticks, minor=False)
    axs[1,0].set_ylabel('WER')
    axs[1,0].set_xlabel('# Epochs')
    axs[1,0].legend()

    axs[1,1].plot(default_x_ticks, bleu, label='BLEU')
    axs[1,1].set_xticks(default_x_ticks, minor=False)
    axs[1,1].set_ylabel('BLEU')
    axs[1,1].set_xlabel('# Epochs')
    axs[1,1].legend()
    plt.show()
