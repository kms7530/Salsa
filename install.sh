# ! /bin/bash
echo "Clone repo from https://github.com/EvolvingLMMs-Lab/LongVA ..."
git clone https://github.com/EvolvingLMMs-Lab/LongVA
cd LongVA

echo "Install dependencies ..."
pip install -e "longva/.[train]"
pip install packaging 
pip install ninja 
pip install flash-attn --no-build-isolation --no-cache-dir
pip install -r requirements.txt

cd ..
echo "Remove unnecessary files ..."
rm -rf ./LongVA
echo "Done!"