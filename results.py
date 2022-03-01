import pickle
from csv import writer
import numpy as np

type = 'accuracies'

# 'Central_Valley', 'Florida', 'Louisiana', 'Tennessee', 'Phoenix'
loc = ['Phoenix']

ps = [224, 448, 672, 896, 1120]

m = ['BYOL', 'BarlowTwins', 'MoCo', 'SimCLR',
     'SimSiam', 'TripletBig', 'Triplet']

clf = ['linear', 'xgboost', 'MLP', 'random_forest']

for location in loc:
    with open(f"{location}_{type}.csv", "w") as empty_csv:
        pass
    for model in m:
        with open(f"{location}_{type}.csv", 'a') as f_object:
            writer_object = writer(f_object)
            #writer_object.writerow([None, model, None, None, None, None, None])
            writer_object.writerow([None, model, None, None, None])
            # writer_object.writerow([None, 'cos positive', 'cos negative',
            #                        'delta cos', 'mse positive', 'mse negative', 'delta mse'])
            writer_object.writerow([None, 'linear', 'XGBoost',
                                    'MLP', 'RF'])
        for patch in ps:
            # with open('/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/dump_pretrained/similarities/'+model+'_similarities_test'+'_'+str(patch)+'_'+location+'.json', 'rb') as data:
            #     d = pickle.load(data)
            # l = [patch,
            #         f"{d['positive']['cos']['standard']['mean']:.3f} +/- {d['positive']['cos']['standard']['std']:.3f}",
            #         f"{d['negative']['cos']['standard']['mean']:.3f} +/- {d['negative']['cos']['standard']['std']:.3f}",
            #         f"{d['positive']['cos']['standard']['mean']-d['negative']['cos']['standard']['mean']:.3f}",
            #         f"{d['positive']['mse']['standard']['mean']:.3f} +/- {d['positive']['mse']['standard']['std']:.3f}",
            #         f"{d['negative']['mse']['standard']['mean']:.3f} +/- {d['negative']['mse']['standard']['std']:.3f}",
            #         f"{d['positive']['mse']['standard']['mean']-d['negative']['mse']['standard']['mean']:.3f}"]
            l = [patch]
            for classifier in clf:
                with open('/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/dump_pretrained/accuracies/'+model+'_'+str(patch)+'_test_accuracies_'+classifier+'.pkl', 'rb') as data:
                    d = pickle.load(data)
                    avg = []
                    for k in d.dict[location]:
                        v = d.dict[location][k]
                        avg.append(v[0])
                    std = np.std(avg)*100
                    avg = np.mean(avg)*100
                l.append(f"{avg:.2f}% +/-{std:.2f}")
            with open(f"{location}_{type}.csv", 'a') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(l)
        with open(f"{location}_{type}.csv", 'a') as f_object:
            writer_object = writer(f_object)
            # writer_object.writerow([None, None, None, None, None, None, None])
            writer_object.writerow([None, None, None, None, None])
