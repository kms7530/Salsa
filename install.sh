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
echo ">>> Install dependencies for Vigilant ..."
echo ""
pip install -r requirements.txt

echo ""
echo ">>> Done!"
echo ""