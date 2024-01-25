import os

benign_path = "benign_25k"
misleading_path = "misleading"
phish_path = "phish_sample_30k"
legitimate = "Legitimate"
phishing = "Phishing"
if not (os.path.exists(legitimate)): os.mkdir(legitimate)
if not (os.path.exists(phishing)): os.mkdir(phishing)

def get_html_txt(source_path, destination_path):
    for folder in os.listdir(source_path):
        if(os.path.exists(os.path.join(source_path, folder, "html.txt"))):
            source = os.path.join(source_path, folder, "html.txt")
            destination = os.path.join(destination_path, folder+".txt")
            os.rename(source, destination)
            
get_html_txt(misleading_path, legitimate)
get_html_txt(benign_path, legitimate)
get_html_txt(phish_path, phishing)