#! /bin/bash

cd tools
mkdir spice
cd spice
wget "https://panderson.me/images/SPICE-1.0.zip" ./
mv SPICE-1.0/* ./
rm -r SPICE-1.0
rm SPICE-1.0.zip
bash get_stanford_models.sh
