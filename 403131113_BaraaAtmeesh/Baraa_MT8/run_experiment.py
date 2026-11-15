from pathlib import Path
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchaudio
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support, confusion_matrix,
                             ConfusionMatrixDisplay)
from src.audio_pipeline import segment_words, spectral_features

ROOT = Path(__file__).resolve().parent


def show_segmentation(signal, sample_rate, spans, info, filename, output):
    fig, ax = plt.subplots(figsize=(11,2.9))
    ax.plot(np.arange(len(signal))/sample_rate,signal,color="#35445c",lw=.65)
    if spans is not None:
        for i,(a,b) in enumerate(spans):
            ax.axvspan(a/sample_rate,b/sample_rate,color="#007f86" if i%2==0 else "#d38a28",alpha=.24)
            ax.text((a+b)/(2*sample_rate),.85*ax.get_ylim()[1],str(i+1),ha="center",fontsize=8)
    else:
        ax.text(.02,.93,info["reason"],transform=ax.transAxes,color="#a32020",va="top")
    ax.set(title=filename,xlabel="Time (s)",ylabel="Amplitude")
    fig.tight_layout()
    fig.savefig(output,dpi=145)
    plt.close(fig)


def bootstrap_group_accuracy(labels, predicted, groups, repeats=1000, seed=2026):
    rng=np.random.default_rng(seed)
    unique=np.unique(groups)
    group_indexes={k:np.flatnonzero(groups==k) for k in unique}
    samples=[]
    for _ in range(repeats):
        chosen=rng.choice(unique,size=len(unique),replace=True)
        indices=np.concatenate([group_indexes[k] for k in chosen])
        samples.append(accuracy_score(labels[indices],predicted[indices]))
    return np.quantile(samples,[.025,.975])


def main():
    for folder in ["data","results","figures"]:
        (ROOT/folder).mkdir(exist_ok=True)
    review=ROOT/"figures"/"segmentation_review"
    review.mkdir(exist_ok=True)
    dataset=torchaudio.datasets.YESNO(root=str(ROOT/"data"),
             url="https://www.openslr.org/resources/1/waves_yesno.tar.gz", download=True)
    audio_files=sorted((ROOT/"data"/"waves_yesno").glob("*.wav"))
    if len(dataset)!=len(audio_files):
        raise RuntimeError("Dataset list and audio files do not match")
    print(f"YESNO loaded through torchaudio.datasets.YESNO: {len(dataset)} files")
    X,y,groups,identifiers=[],[],[],[]
    rejected=[]; accepted=[]
    for file_index in range(len(dataset)):
        waveform,sample_rate,labels=dataset[file_index]
        filename=audio_files[file_index].name
        expected=[int(part) for part in audio_files[file_index].stem.split("_")]
        if list(labels)!=expected:
            raise ValueError(f"Label order does not match filename: {filename}")
        signal=waveform.mean(dim=0).numpy()
        spans,info=segment_words(signal,sample_rate)
        show_segmentation(signal,sample_rate,spans,info,filename,review/f"{file_index:02d}.png")
        if spans is None:
            rejected.append({"file":filename,"reason":info["reason"]})
            continue
        if len(spans)!=len(labels):
            raise ValueError(f"Alignment mismatch in {filename}")
        accepted.append(filename)
        for word_index,((start,end),label) in enumerate(zip(spans,labels),1):
            X.append(spectral_features(signal[start:end],sample_rate))
            y.append(int(label));groups.append(file_index)
            identifiers.append(f"{filename}:word{word_index}")
    pd.DataFrame(rejected,columns=["file","reason"]).to_csv(ROOT/"results"/"rejected_recordings.csv",index=False)
    X=np.asarray(X);y=np.asarray(y);groups=np.asarray(groups)
    np.savez_compressed(ROOT/"results"/"features.npz",X=X,y=y,groups=groups,ids=identifiers)
    assert len(X)==len(y)==len(groups) and X.shape[1]==40
    if len(np.unique(groups))<16:
        raise RuntimeError("Not enough reliably segmented recordings")
    train_idx,test_idx=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=2026).split(X,y,groups))
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
    cv=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=2026)
    models={
        "Linear SVM":make_pipeline(StandardScaler(),SVC(kernel="linear",C=0.5)),
        "MLP Neural Network":make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(32,16),alpha=.01,max_iter=1100,random_state=2026)),
        "Extra Trees":make_pipeline(StandardScaler(),ExtraTreesClassifier(n_estimators=160,min_samples_leaf=2,random_state=2026,n_jobs=-1)),
        "KNN (k=7)":make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=7,weights="distance")),
        "Gaussian Naive Bayes":make_pipeline(StandardScaler(),GaussianNB(var_smoothing=1e-8))
    }
    scores=[]; predictions={}
    for name,model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",category=UserWarning)
            cv_scores=cross_val_score(model,X[train_idx],y[train_idx],groups=groups[train_idx],cv=cv,scoring="balanced_accuracy")
            model.fit(X[train_idx],y[train_idx])
        predicted=model.predict(X[test_idx]);predictions[name]=predicted
        precision,recall,f1,_=precision_recall_fscore_support(y[test_idx],predicted,average="binary",pos_label=1,zero_division=0)
        lo,hi=bootstrap_group_accuracy(y[test_idx],predicted,groups[test_idx])
        scores.append({"Model":name,"CV balanced accuracy mean":float(np.mean(cv_scores)),
         "CV balanced accuracy std":float(np.std(cv_scores,ddof=1)),
         "Test accuracy":accuracy_score(y[test_idx],predicted),
         "Test balanced accuracy":balanced_accuracy_score(y[test_idx],predicted),
         "YES precision":precision,"YES recall":recall,"YES F1":f1,
         "Errors":int(np.sum(predicted!=y[test_idx])),"Accuracy 95pct bootstrap low":lo,
         "Accuracy 95pct bootstrap high":hi})
        print(f"{name:23s} Test accuracy: {scores[-1]['Test accuracy']:.4f}; train CV balanced accuracy: {cv_scores.mean():.4f}")
    table=pd.DataFrame(scores).sort_values("Test balanced accuracy",ascending=False)
    table.to_csv(ROOT/"results"/"model_comparison.csv",index=False,float_format="%.6f")
    pd.DataFrame({"word":np.asarray(identifiers)[test_idx],"recording_group":groups[test_idx],
      "true_label":y[test_idx],**{f"prediction_{i+1}":p for i,p in enumerate(predictions.values())}}).to_csv(ROOT/"results"/"test_predictions.csv",index=False)
    meta={"recordings_total":len(dataset),"recordings_accepted":len(accepted),
      "recordings_excluded":len(rejected),"words_used":len(y),"yes_words":int(np.sum(y)),
      "no_words":int(len(y)-np.sum(y)),"train_words":len(train_idx),"test_words":len(test_idx),
      "train_recordings":len(np.unique(groups[train_idx])),"test_recordings":len(np.unique(groups[test_idx])),
      "feature_dimension":X.shape[1],"cv_folds":4,"seed":2026,
      "segmentation_note":"Energy-peak selection was checked against visualization; the unresolved recording was excluded."}
    (ROOT/"results"/"run_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(9,4.2))
    ax.barh(table["Model"],table["Test balanced accuracy"],color="#197e88")
    ax.set(xlim=(0,1.08),xlabel="Test balanced accuracy",title="YESNO: independent recording test set")
    ax.invert_yaxis()
    for row,v in enumerate(table["Test balanced accuracy"]):
        ax.text(min(1.02,v+.012),row,f"{v:.3f}",va="center",fontsize=9)
    fig.tight_layout();fig.savefig(ROOT/"figures"/"model_comparison.png",dpi=175);plt.close(fig)
    fig,axes=plt.subplots(1,len(models),figsize=(16,3.7))
    for ax,(name,pred) in zip(axes,predictions.items()):
        matrix=confusion_matrix(y[test_idx],pred,labels=[0,1])
        ConfusionMatrixDisplay(matrix,display_labels=["NO","YES"]).plot(ax=ax,cmap="Blues",colorbar=False)
        ax.set_title(name,fontsize=9)
    fig.tight_layout();fig.savefig(ROOT/"figures"/"confusion_matrices.png",dpi=170);plt.close(fig)
    ablations=[]
    for subset,cols in [("Log-frequency band statistics",np.arange(32)),
                        ("Spectral descriptors only",np.arange(32,40)),("All 40 features",np.arange(40))]:
        for name,base in [("Linear SVM",SVC(kernel="linear",C=.5)),
                          ("Extra Trees",ExtraTreesClassifier(n_estimators=160,min_samples_leaf=2,random_state=2026,n_jobs=-1))]:
            model=make_pipeline(StandardScaler(),base)
            model.fit(X[train_idx][:,cols],y[train_idx]);pred=model.predict(X[test_idx][:,cols])
            ablations.append({"Feature set":subset,"Model":name,
                "Test balanced accuracy":balanced_accuracy_score(y[test_idx],pred)})
    ablation_table=pd.DataFrame(ablations)
    ablation_table.to_csv(ROOT/"results"/"feature_ablation.csv",index=False)
    fig,ax=plt.subplots(figsize=(8.8,4.3))
    ablation_table.pivot(index="Feature set",columns="Model",values="Test balanced accuracy").plot(
        kind="bar",ax=ax,rot=6,ylim=(0,1.09),color=["#197e88","#e8a650"])
    ax.set(title="Feature comparison on the same test recordings",xlabel="",ylabel="Balanced accuracy")
    fig.tight_layout();fig.savefig(ROOT/"figures"/"feature_ablation.png",dpi=175);plt.close(fig)
    print("Saved new results, plots and segmentation review images")
    print(json.dumps(meta,indent=2))

if __name__=="__main__":
    main()
