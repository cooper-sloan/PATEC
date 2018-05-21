import subprocess

nb_teachers=100
dataset="mnist"

for i in range(nb_teachers):
    subprocess.run(
        ["python3", "train_teachers.py", "--nb_teachers="+str(nb_teachers), "--teacher_id="+str(i), "--dataset="+dataset])
    print('-'*80)

print("All Teachers Trained")
