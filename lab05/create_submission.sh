cp mean-shift/mean_shift.py mean-shift/mean-shift.py

zip -r submission_armstrong.zip \
    mean-shift/mean-shift.py \
    seg-net/lib/models/seg_net_lite.py \
    seg-net/out/model_best.pth.tar \
    writeup/main.pdf \
    -x "*/__pycache__/*"

rm mean-shift/mean-shift.py
