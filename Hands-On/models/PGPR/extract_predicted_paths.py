import csv

def save_pred_paths(folder_path, pred_paths, train_labels):
    print("Normalizing items scores...")
    #Get min and max score to performe normalization between 0 and 1
    score_list = []
    for uid, pid in pred_paths.items():
        for pid, path_list in pred_paths[uid].items():
            if pid in set(train_labels[uid]): continue
            for path in path_list:
                score_list.append(float(path[0]))
    min_score = min(score_list)
    max_score = max(score_list)

    print("Saving pred_paths...")
    with open(folder_path + "/pred_paths.csv", 'w+', newline='') as pred_paths_file:
        header = ["uid", "pid", "path_score", "path_prob", "path"]
        writer = csv.writer(pred_paths_file)
        writer.writerow(header)
        for uid, pid in pred_paths.items():
            for pid, path_list in pred_paths[uid].items():
                if pid in set(train_labels[uid]): continue
                for path in path_list:
                    path_score = str((float(path[0]) - min_score) / (max_score - min_score))
                    path_prob = path[1]
                    path_explaination = []
                    for tuple in path[2]:
                        for x in tuple:
                            path_explaination.append(str(x))
                    writer.writerow([uid, pid, path_score, path_prob, ' '.join(path_explaination)])
    pred_paths_file.close()

def save_best_pred_paths(folder_path, best_pred_paths):
    print("Normalizing items scores...")
    # Get min and max score to performe normalization between 0 and 1
    score_list = []
    for uid, pid_list in best_pred_paths.items():
        for path in pid_list:
            score_list.append(path[0])
    min_score = min(score_list)
    max_score = max(score_list)

    print("Saving best_pred_paths...")
    with open(folder_path + "/best_pred_paths.csv", 'w+', newline='') as best_pred_paths_file:
        header = ["uid", "rec item", "path_score", "path_prob", "path"]
        writer = csv.writer(best_pred_paths_file)
        writer.writerow(header)
        for uid, recs in best_pred_paths.items():
            for rec in recs:
                path_score = str((rec[0] - min_score) / (max_score - min_score))
                path_prob = rec[1]
                recommended_item_id = rec[2][-1][-1]
                path_explaination = []
                for tuple in rec[2]:
                    for x in tuple:
                        path_explaination.append(str(x))
                writer.writerow([uid, recommended_item_id, path_score, path_prob, ' '.join(path_explaination)])
    best_pred_paths_file.close()

def save_pred_labels(folder_path, pred_labels):
    print("Saving topks...")
    with open(folder_path +  "/uid_topk.csv", 'w+', newline='') as uid_topk:
        header = ["uid", "top10"]
        writer = csv.writer(uid_topk)
        writer.writerow(header)
        for uid, topk in pred_labels.items():
            writer.writerow([uid, ' '.join([str(x) for x in topk[::-1]])])
    uid_topk.close()

def save_pred_explainations(folder_path, pred_paths_top10, pred_labels):
    print("Saving topks' explanations")
    # Save explainations to load the uid-pid selected explaination
    with open(folder_path + "/uid_pid_explanation.csv", 'w+', newline='') as uid_pid_explaination:
          header = ["uid", "pid", "path"]
          writer = csv.writer(uid_pid_explaination)
          writer.writerow(header)
          for uid, paths in pred_paths_top10.items():
              for idx, path in enumerate(paths[::-1]):
                  path_explaination = []
                  for tuple in path:
                      for x in tuple:
                          path_explaination.append(str(x))
                  writer.writerow([uid, pred_labels[uid][idx], ' '.join(path_explaination)])
    uid_pid_explaination.close()
