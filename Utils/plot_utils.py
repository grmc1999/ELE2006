import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
def add_label(violin, label,labels):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))

def plot_distances(distances_dictionary):
    Model_name=list(distances_dictionary["Train"].keys())[0].split("_")[0]
    plt.figure(figsize=(20,10))
    plt.title(Model_name)
    
    train_data=distances_dictionary["Train"]
    validation_data=distances_dictionary["Test"]
    for i,model_name in enumerate(distances_dictionary["Train"].keys()):
        i=i*2
        print(i,model_name)
        plt.scatter(y=train_data[model_name]["shared"] ,
                    x=i*np.ones(train_data[model_name]["shared"][0].shape)-0.5,alpha=0.1,s=150.0,color="r",
                    )
        plt.scatter(y=validation_data[model_name]["shared"] ,
                    x=i*np.ones(validation_data[model_name]["shared"][0].shape)-0.25,alpha=0.1,s=150.0,color="b",
                    label="test_shared_space")
        plt.scatter(y=train_data[model_name]["privated"] ,
                    x=0.25+i*np.ones(train_data[model_name]["privated"][0].shape),alpha=0.1,s=150.0,color="r",
                    label="train_privated_space")
        plt.scatter(y=validation_data[model_name]["privated"] ,
                    x=0.5+i*np.ones(validation_data[model_name]["privated"][0].shape),alpha=0.1,s=150.0,color="b",
                   label="test_privated_space")
    labels = []
    vts=plt.violinplot(
        [distances_dictionary['Train'][k]["shared"][0] for k in distances_dictionary['Train'].keys()],
        positions=np.array(list(range(0,10,2)))-0.5,showmeans=True)
    add_label(vts,"train_shared_space",labels)
    vvs=plt.violinplot(
        [distances_dictionary['Test'][k]["shared"][0] for k in distances_dictionary['Test'].keys()],
        positions=np.array(list(range(0,10,2)))-0.25,showmeans=True)
    add_label(vvs,"test_shared_space",labels)
    vtp=plt.violinplot(
        [distances_dictionary['Train'][k]["privated"][0] for k in distances_dictionary['Train'].keys()],
        positions=np.array(list(range(0,10,2)))+0.25,showmeans=True)
    add_label(vtp,"train_privated_space",labels)
    vvp=plt.violinplot(
        [distances_dictionary['Test'][k]["privated"][0] for k in distances_dictionary['Test'].keys()],
        positions=np.array(list(range(0,10,2)))+0.5,showmeans=True)
    add_label(vvp,"test_privated_space",labels)
    
    plt.legend(*zip(*labels), loc=2)
    plt.xticks(list(range(0,10,2)),list(distances_dictionary["Train"].keys()))
    return plt


def plot_multiple_models(distances_dictionaries,figure,distance_name):
    
    plt=figure
    plt.title(distance_name)
    
    labels = []
    for j,(distances_dictionary,name) in enumerate(zip(distances_dictionaries,["MT2PA","PA2MA"])):
        train_data=distances_dictionary["Train"]
        validation_data=distances_dictionary["Test"]
        
        
        for i,model_name in enumerate(distances_dictionary["Train"].keys()):
            i=i*2+10*j
            print(i,model_name)
            plt.scatter(y=train_data[model_name]["shared"] ,
                        x=i*np.ones(train_data[model_name]["shared"][0].shape)-0.5,alpha=0.1,s=50.0,color="r",
                        )
            plt.scatter(y=validation_data[model_name]["shared"] ,
                        x=i*np.ones(validation_data[model_name]["shared"][0].shape)-0.25,alpha=0.1,s=50.0,color="b",
                        label="test_shared_space")
            plt.scatter(y=train_data[model_name]["privated"] ,
                        x=0.25+i*np.ones(train_data[model_name]["privated"][0].shape),alpha=0.1,s=50.0,color="r",
                        label="training_privated_space")
            plt.scatter(y=validation_data[model_name]["privated"] ,
                        x=0.5+i*np.ones(validation_data[model_name]["privated"][0].shape),alpha=0.1,s=50.0,color="b",
                       label="test_privated_space")
        
        ranges=(10*j,10*(j+1),2)
        vts=plt.violinplot(
            [distances_dictionary['Train'][k]["shared"][0] for k in distances_dictionary['Train'].keys()],
            positions=np.array(list(range(*ranges)))-0.5,showmeans=True)
        add_label(vts,"training_shared_space_"+name,labels)
        vvs=plt.violinplot(
            [distances_dictionary['Test'][k]["shared"][0] for k in distances_dictionary['Test'].keys()],
            positions=np.array(list(range(*ranges)))-0.25,showmeans=True)
        add_label(vvs,"test_shared_space_"+name,labels)
        vtp=plt.violinplot(
            [distances_dictionary['Train'][k]["privated"][0] for k in distances_dictionary['Train'].keys()],
            positions=np.array(list(range(*ranges)))+0.25,showmeans=True)
        add_label(vtp,"training_privated_space_"+name,labels)
        vvp=plt.violinplot(
            [distances_dictionary['Test'][k]["privated"][0] for k in distances_dictionary['Test'].keys()],
            positions=np.array(list(range(*ranges)))+0.5,showmeans=True)
        add_label(vvp,"test_privated_space_"+name,labels)

    plt.legend(*zip(*labels), loc=2)
    plt.xticks(
        list(range(0,20,2)),
        list(distances_dictionaries[0]["Train"].keys())+list(distances_dictionaries[1]["Train"].keys()))
    return plt