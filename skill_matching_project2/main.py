import os
import pandas as pd
import pypdf
import pickle
import re
from sentence_transformers import SentenceTransformer, util
import torch
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields, abort
import werkzeug
import numpy as np

app = Flask(__name__)
api = Api(app)

# Define API models
recommendation_fields = api.model('Recommendations', {
    'Skill': fields.String(description="The skill name."),
    'Score': fields.Float(description="The match score (cosine similarity)."),
})

response_fields = api.model('Response', {
    'recommendations': fields.List(fields.Nested(recommendation_fields),
                                   description="Skill names and match score for the recommended skills. The recommendations are in descending order by score.")
})

# Initialize SentenceTransformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# API parser for file upload
file_upload_parser = api.parser()
file_upload_parser.add_argument('file', location='files', type=werkzeug.datastructures.FileStorage, required=True)

def read_pdf_text(file_path):
    pdf_reader = pypdf.PdfReader(file_path)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def cut_and_clean(text):
    snippets = re.split(r'\n+', text)
    snippets = [snippet.strip() for snippet in snippets if snippet.strip()]
    return snippets

def match_snippets(snippets, master_phrase_embs, master_phrase_list, top_k=5):
    snippet_embs = embedder.encode(snippets, convert_to_tensor=True, device=device)
    cos_scores = util.pytorch_cos_sim(snippet_embs, master_phrase_embs)
    
    top_k = min(top_k, len(master_phrase_list))  # Limit top_k to the number of available skills
    top_results = torch.topk(cos_scores, k=top_k, dim=1)

    recommendations = []
    for idx, snippet in enumerate(snippets):
        for score, idx in zip(top_results.values.cpu().numpy()[idx], top_results.indices.cpu().numpy()[idx]):
            recommendations.append({'Skill': master_phrase_list[idx], 'Score': score, 'Snippet': snippet})
    
    return pd.DataFrame(recommendations)

@api.route('/skills_from_cv')
class SkillsFromCV(Resource):
    @api.marshal_with(response_fields, as_list=True)
    @api.expect(file_upload_parser)
    def post(self):
        args = file_upload_parser.parse_args()
        input_file = args['file']
        input_file.save('file.pdf')

        master_skills_emb_binary = './master_emb_list.pkl'
        master_skills_list = './master_skills_list.txt'

        with open(master_skills_emb_binary, 'rb') as f:
            master_phrase_embs = pickle.load(f)
        with open(master_skills_list, 'r') as f:
            lines = f.readlines()
            master_phrase_list = [l.replace("\n", "") for l in lines]

        master_phrase_embs = torch.tensor(master_phrase_embs).to(device)

        file_text = read_pdf_text('file.pdf')
        cv_snippets = cut_and_clean(file_text)
        skill_recommendation = match_snippets(cv_snippets, master_phrase_embs, master_phrase_list)

        skill_recommendation = skill_recommendation[skill_recommendation['Score'] >= 0.3]
        skill_recommendation = skill_recommendation.sort_values('Score', ascending=False)
        skill_recommendation = skill_recommendation.drop_duplicates(subset='Skill').reset_index(drop=True)

        response = {'recommendations': skill_recommendation.to_dict(orient='records')}
        return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
