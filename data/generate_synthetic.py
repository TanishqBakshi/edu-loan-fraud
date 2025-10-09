# data/generate_synthetic.py
import csv, random, os

OUTPUT = os.path.join(os.path.dirname(__file__), "sample_labeled.csv")
fields = [
    'applicant_id','avg_hold_time','std_hold','backspace_freq',
    'interkey_mean','mouse_jitter','ocr_confidence','device_fp_sim','label'
]

def gen_row(i):
    aid = f'app_{i}'
    avg_hold = max(20, random.gauss(120, 30))
    std_hold = max(5, abs(random.gauss(40, 15)))
    backspace = round(random.random(), 3)
    interkey = max(20, random.gauss(150, 50))
    jitter = round(random.random() * 5, 3)
    ocr = random.randint(60, 99)
    dsim = round(random.random(), 3)
    label = 1 if (ocr < 70 or dsim < 0.2 or avg_hold < 50) else 0
    return [aid, round(avg_hold,3), round(std_hold,3), backspace, round(interkey,3), jitter, ocr, dsim, label]

def main(n=1200):
    with open(OUTPUT, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i in range(n):
            w.writerow(gen_row(i))
    print(f'✅ wrote {OUTPUT}')

if __name__ == "__main__":
    main()
