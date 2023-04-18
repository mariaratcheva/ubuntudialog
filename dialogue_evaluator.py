
import torch
from transformers import BertModel, BertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to(device)

class DialogueEvaluator():
  '''
    This class evaluates the performance of a dialogue model using embedding metrics.
    
    Arguments
    ---------
    tokenizer : torch.nn.Module
        The tokenizer to be used to encode the text.
    model : torch.nn.Module
        The model to be used to exttract the word embeddings.

    Example
    -------
  '''
  def __init__(self, tokenizer, model):
    self.model = model
    self.tokenizer = tokenizer

  def get_embeddings(self, text):
    ''' Extracts the embeddings of a given text according to provided model.
    
    Arguments
    ---------
    text : string
        the text for which we need the representation

    Return
    ------
    embeddings : torch.Tensor
        the word embeddings representation of the text

    len_input_ids: int
        the number of tokens of the encoded text

    Example
    -------
    '''
    # tokenize the text with the corresponding tokenizer
    input_ids = self.tokenizer.encode(text, add_special_tokens=True, return_tensors='pt').to(device)

    # get the last hidden state of the model
    # as an alternative some articles recommend to sum up the embedding of the last 4 layers
    with torch.no_grad():
        outputs = self.model(input_ids)
        last_hidden_state = outputs.last_hidden_state

    # return the embeddings at the last hidden state
    embeddings = last_hidden_state
    return embeddings, len(input_ids[0])


  def extrema_embedding_score(self, prediction, ground_truth):
    ''' Computes the extrema embedding score/distiance 
        between the prediction text and ground truth  text.
    '''
    # compute the embeddings 
    pred_embeddings, lens_pred = self.get_embeddings(prediction)
    gt_embeddings, lens_gt = self.get_embeddings(ground_truth)

    # compute the extrema vector
    pred_extrema_vector, pred_max_idxs = torch.max(pred_embeddings.abs(), dim=-1)
    gt_extrema_vector, gt_max_idxs = torch.max(gt_embeddings.abs(), dim=-1)

    # remove the batch dimension
    pred_extrema_vector = pred_extrema_vector.squeeze(dim=0)
    gt_extrema_vector = gt_extrema_vector.squeeze(dim=0)

    # add padding if necessary
    if (lens_pred - lens_gt !=0):
      to_pad = np.abs(lens_gt - lens_pred)
      padding = (0,to_pad)
      pad = torch.nn.ZeroPad2d(padding)

      # pad the tensor
      if lens_pred > lens_gt:
        gt_extrema_vector = pad(gt_extrema_vector)
      else:
        pred_extrema_vector = pad(pred_extrema_vector)

    # compute the cosine similarity between the extrema vectors
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    score = cos(pred_extrema_vector, gt_extrema_vector)

    return score.item()

  def greedy_embedding_score(self, prediction, ground_truth):
    ''' Computes the greedy embedding score/distiance 
        between the prediction text and ground truth text.
    '''
    # Compute the embeddings 
    pred_embeddings, lens_pred = self.get_embeddings(prediction)
    gt_embeddings, lens_gt = self.get_embeddings(ground_truth)

    #remove the batch dimension
    pred_embeddings = pred_embeddings.squeeze(dim=0)
    gt_embeddings = gt_embeddings.squeeze(dim=0)

    # For each token in the generated text, 
    # compute the cosine similarity between its embedding and the embeddings 
    # of all the tokens in the reference text.
    n1 = lens_pred
    n2 = lens_gt
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cosine_similarities = [cos(pred_embeddings[i,:], gt_embeddings[j,:]).item() for i in range(n1) for j in range(n2)]
    cosine_similarities = torch.tensor(cosine_similarities).view(n1, n2)

    # Select the reference token with the highest cosine similarity for each generated token, 
    # and sum the cosine similarity scores for all the pairs of matched tokens.
    matched_indices = cosine_similarities.argmax(axis=1)
    matched_scores = cosine_similarities.max(axis=1).values

    # Compute the average greedy embedding score
    score = matched_scores.sum() / lens_pred
    return score.item()


  def average_embedding_score(self, prediction, ground_truth):
    ''' Computes the average embedding score/distiance 
          between the prediction text and ground truth text.
    '''
    # compute the embeddings 
    pred_embeddings, lens_pred = self.get_embeddings(prediction)
    gt_embeddings, lens_gt = self.get_embeddings(ground_truth)

    # remove the batch dimension
    pred_embeddings = pred_embeddings.squeeze(dim=0)
    gt_embeddings = gt_embeddings.squeeze(dim=0)

    # calculate the average similarity score across all pairs of embeddings. 
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cosine_similarities = [cos(pred_embeddings[i,:], gt_embeddings[i,:]).item() for i in range(lens_pred)]
    
    # compute the average embedding score
    cosine_similarities = torch.tensor(cosine_similarities)
    score = cosine_similarities.sum() / lens_pred
    return score.item()
