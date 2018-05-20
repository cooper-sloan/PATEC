import subprocess
import deep_recommender

nb_teachers=10
dataset="netflix"

for i in range(nb_teachers):
    print("Training Teacher %s" % i)
    deep_recommender.train_teacher(nb_teachers, i)
    print('-'*160)

print("All Teachers Trained")
