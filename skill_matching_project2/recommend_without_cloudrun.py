
import os
import sys
import pandas as pd
import pypdf
import pickle
import re
from sentence_transformers import SentenceTransformer, util
import torch
import werkzeug
import numpy as np


# Use a pre-trained model from Hugging Face
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Check if a GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedder.to(device)

def read_pdf_text(pdf_path):
    '''
    Parses a PDF file and returns the contents.
    '''
    file_text = ''
    with open(pdf_path, 'rb') as f:
        pdf = pypdf.PdfReader(f)
        for page in range(len(pdf.pages)):
            file_text += (pdf.pages[page].extract_text())
    return file_text


def cut_and_clean(string):
    '''
    Cut up text into smaller pieces for the model to read and clean the pieces.
    '''
    chunks = re.split(r'\n|\.', string)
    chunks = [x for x in chunks if len(x) > 4]
    c_chunks = list()
    for i in chunks:
        i = ''.join((x for x in i if not x.isdigit()))  # throw away digits
        i = re.sub(r'[^a-zA-Z0-9 \n\.,]', ' ', i)  # throw away special characters
        i = " ".join(i.split())  # remove extra spaces
        i = i.lower()  # lowercase
        if len(i.split()) > 3:
            c_chunks.append(i)
    return c_chunks


def match_snippets(snippets, master_phrase_embs, master_phrase_list, top_k):
    '''
    Match a list of short phrases to a set of phrase embeddings.
    '''
    skill_recommendation = pd.DataFrame()
    for query in snippets:
        query_embedding = embedder.encode(query.strip(), convert_to_tensor=True, device=device)
        cos_scores = util.cos_sim(query_embedding, master_phrase_embs)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        skills_list = list()
        score_list = list()
        for score, idx in zip(top_results.values.cpu().numpy(), top_results.indices.cpu().numpy()):
            skills_list.append(master_phrase_list[idx])
            score_list.append(score.item())
        
        skills_df = pd.DataFrame(skills_list)
        score_df = pd.DataFrame(score_list)
        sk_sc_df = pd.concat([skills_df, score_df], axis=1)
        sk_sc_df.columns = ['Phrase', 'Score']
        skill_recommendation = pd.concat([skill_recommendation, sk_sc_df]).reset_index(drop=True)
    
    return skill_recommendation



def main(input_file, master_skills_emb_binary, master_skills_list, top_k):

    '''
    Save a set of suggestions for skills from a CV.
    '''
    with open(master_skills_emb_binary, 'rb') as f:
        master_phrase_embs = pickle.load(f)
    
    # Convert numpy array to PyTorch tensor and move to the appropriate device
    master_phrase_embs = torch.tensor(master_phrase_embs).to(device)
    
    with open(master_skills_list, 'r') as f:
        lines = f.readlines()
        master_phrase_list = [l.replace("\n", "") for l in lines]
    
    file_text = read_pdf_text(input_file)
    cv_snippets = cut_and_clean(file_text)
    skill_recommendation = match_snippets(cv_snippets, master_phrase_embs, master_phrase_list, top_k=top_k)
    skill_recommendation = skill_recommendation[skill_recommendation['Score'] >= 0.5]
    skill_recommendation = skill_recommendation.sort_values('Score', ascending=False)
    skill_recommendation = skill_recommendation.drop_duplicates(subset='Phrase').reset_index(drop=True)
    skill_recommendation = skill_recommendation.rename(columns={'Phrase': 'Skill'})
    skill_recommendation.to_csv(os.path.splitext(input_file)[0] + '_skill_suggestions.csv', index=False)


if __name__ == "__main__":
    if 'ipykernel_launcher.py' in sys.argv[0]:
        sys.argv = [arg for arg in sys.argv if not arg.endswith('.json')]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--master_skills_emb_binary', required=False, default=r'./master_emb_list.pkl')
    parser.add_argument('--master_skills_list', required=False, default=r'./master_skills_list.txt')
    parser.add_argument('--top_k', required=False, default=5, type=int)  # Ensure top_k is parsed as an integer
    args = parser.parse_args()
    main(input_file=args.input_file, master_skills_emb_binary=args.master_skills_emb_binary, master_skills_list=args.master_skills_list, top_k=args.top_k)
