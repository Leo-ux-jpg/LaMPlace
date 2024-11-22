import subprocess
import argparse
import os

def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    listOfFile.sort()
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

def main(id,a1,a2,a3,a4,seed):
    benchmark = f"superblue{id}"

    command = ["python", "graph_l_flow.py","--id",id,"--a1",a1,"--a2",a2,"--a3",a3,"--a4",a4]
    subprocess.run(command)
    name = f"{a1}{a2}{a3}{a4}"
    command = ["python", "lamplace.py","--dataset",benchmark,"--seed",seed,"--name",name]
    subprocess.run(command)
    dirName = f'results_lamplace/{benchmark}_{seed}_{name}/pl'
    mp_file = getListOfFiles(dirName)
    if "-" in mp_file[1]:
        mp_file.reverse()
    result = f"graph_data/final_result/{benchmark}_{seed}_{name}.txt"
    res = "graph_data/final_result"
    if not os.path.exists(res):
        os.makedirs(os.path.join(res, f"pl"))
    if os.path.exists(result):
        with open(result, 'w') as file:
            file.write('')
    for i,file in enumerate(mp_file):
        command = ["python", "dreamplace/Placer_rp.py","--config",f"test/iccad2015.ot/{benchmark}.json","--pl",file,"--res_dir",result]
        subprocess.run(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="1")
    parser.add_argument("--a1", default="0.25")
    parser.add_argument("--a2", default="0.25")
    parser.add_argument("--a3", default="0.25")
    parser.add_argument("--a4", default="0.25")
    parser.add_argument("--seed", default="2023")
    args = parser.parse_args()
    id = args.id
    a1 = args.a1
    a2 = args.a2
    a3 = args.a3
    a4 = args.a4
    main(id,a1,a2,a3,a4,args.seed)