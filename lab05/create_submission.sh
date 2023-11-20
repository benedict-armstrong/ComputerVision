cp mean-shift/mean_shift.py mean-shift/mean-shift.py

zip -r submission_armstrong.zip \
    mean-shift/mean-shift.py \
    seg-net/lib/models/seg_net_lite.py \
    writeup/main.pdf \
    \
    -x "*/__pycache__/*" # model_best.pth.tar \

rm mean-shift/mean-shift.py
