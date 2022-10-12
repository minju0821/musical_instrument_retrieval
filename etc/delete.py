from glob import glob
import os


if __name__ == "__main__":
    
    dir_list = glob("/disk2/aiproducer_inst/rendered_single_inst/test/*")

    for dir in dir_list:
        file_list = glob(dir + "/*.wav")
        file_list = sorted(file_list)
        for file in file_list:
            num = int(file.split("/")[-1].replace('.wav', ''))

            if num > 1000:
                print(file)
                os.remove(file)

        
        