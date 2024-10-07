# ! /bin/bash
echo ""
echo ">>> Clone repo from https://github.com/EvolvingLMMs-Lab/LongVA ..."
echo ""
git clone https://github.com/EvolvingLMMs-Lab/LongVA
cd LongVA

echo ""
echo ">>> Install dependencies for LongVA ..."
echo ""
pip install -e "longva/.[train]"
pip install packaging 
pip install ninja 
pip install flash-attn --no-build-isolation --no-cache-dir
pip install -r requirements.txt

echo ""
echo ">>> Clone repo from https://github.com/IDEA-Research/GroundingDINO.git ..."
echo ""
cd ..
git clone https://github.com/IDEA-Research/GroundingDINO.git

echo ""
echo ">>> Install dependencies for Grounding DINO ..."
echo ""
cd GroundingDINO/
pip install -e .

echo ""
echo ">>> Download Grouding DINO weight ..."
echo ""
mkdir weights
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..

echo ""
echo ">>> Install dependencies for Vigilant ..."
echo ""
cd ..
pip install -r requirements.txt
pip install -U transformers

echo ""
echo ">>> Copy config files ..."
echo ""
cp config/config_template.py ./config.py
cp config/bentofile.yaml ./bentofile.yaml

echo ""
echo ">>> Done!"
echo ""