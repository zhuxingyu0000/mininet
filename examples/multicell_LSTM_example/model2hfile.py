#默认模型文件位置，可修改
MODEL_DIR="model_dir/"
MODEL_META_NAME="MyModel.meta"

#默认配置文件目录
CONFIG_FILE="netmodel.json"

#默认框架
DEFAULT_FRAMEWORK="tensorflow"

#默认文件输出目录
OUT_DICTIONARY="c_sourcefile/weights/"

print("默认模型文件目录："+MODEL_DIR)
print("默认meta文件名："+MODEL_META_NAME)
print("默认配置文件目录："+CONFIG_FILE)
print("Use "+DEFAULT_FRAMEWORK)

if DEFAULT_FRAMEWORK=="tensorflow":
    import tensorflow as tf
    import numpy as np
    import json
    weights=[]
    LSTM_w=[]
    with open(CONFIG_FILE, "r", encoding='utf-8') as f:
        infile=json.load(f)
        for i in infile["nets"]:
            if "var" in i.keys() and i['id']!='LSTM':
                weights.append(i["var"])
            if i['id']=='LSTM':
                LSTM_w.append(i["var"])
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(MODEL_DIR+MODEL_META_NAME)
        saver.restore(sess,tf.train.latest_checkpoint(MODEL_DIR))
        for i in weights:
            print(i+" "+str(np.array(sess.run(tf.get_collection(i))).shape))
            arr=np.array(sess.run(tf.get_collection(i))).flatten()
            with open(OUT_DICTIONARY+i+".h","wb") as f:
                f.write(str("float "+i+"_w[]={\n").encode('utf-8'))
                for x in arr:
                    f.write(str(str(x)+", ").encode('utf-8'))
                f.seek(-2,1)
                f.write(str("};").encode('utf-8'))
        for i in LSTM_w:
            for j in i:
                print(j+" "+str(np.array(sess.run(tf.get_collection(j))).shape))
                arr=np.array(sess.run(tf.get_collection(j)))
                #print(j+"  "+str(arr))
                I=[]
                C=[]
                F=[]
                O=[]
                if len(arr.shape)==2:
                    I=arr[:,0:int(arr.shape[1]/4)]
                    C=arr[:,int(arr.shape[1]/4):int(arr.shape[1]/2)]
                    F=arr[:,int(arr.shape[1]/2):int(arr.shape[1]/2+arr.shape[1]/4)]
                    O=arr[:,int(arr.shape[1]/2+arr.shape[1]/4):int(arr.shape[1])]
                if len(arr.shape)==3:
                    I=arr[:,:,0:int(arr.shape[2]/4)]
                    C=arr[:,:,int(arr.shape[2]/4):int(arr.shape[2]/2)]
                    F=arr[:,:,int(arr.shape[2]/2):int(arr.shape[2]/2+arr.shape[2]/4)]
                    O=arr[:,:,int(arr.shape[2]/2+arr.shape[2]/4):int(arr.shape[2])]
                I=np.array(I).flatten()
                C=np.array(C).flatten()
                F=np.array(F).flatten()
                O=np.array(O).flatten()
                with open(OUT_DICTIONARY+j+"_I.h","wb") as f:
                    f.write(str("float "+j+"_I[]={\n").encode('utf-8'))
                    for x in I:
                        f.write(str(str(x)+", ").encode('utf-8'))
                    f.seek(-2,1)
                    f.write(str("};").encode('utf-8'))
                with open(OUT_DICTIONARY+j+"_C.h","wb") as f:
                    f.write(str("float "+j+"_C[]={\n").encode('utf-8'))
                    for x in C:
                        f.write(str(str(x)+", ").encode('utf-8'))
                    f.seek(-2,1)
                    f.write(str("};").encode('utf-8'))
                with open(OUT_DICTIONARY+j+"_F.h","wb") as f:
                    f.write(str("float "+j+"_F[]={\n").encode('utf-8'))
                    for x in F:
                        f.write(str(str(x)+", ").encode('utf-8'))
                    f.seek(-2,1)
                    f.write(str("};").encode('utf-8'))
                with open(OUT_DICTIONARY+j+"_O.h","wb") as f:
                    f.write(str("float "+j+"_O[]={\n").encode('utf-8'))
                    for x in O:
                        f.write(str(str(x)+", ").encode('utf-8'))
                    f.seek(-2,1)
                    f.write(str("};").encode('utf-8'))
