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
    with open(CONFIG_FILE, "r", encoding='utf-8') as f:
        infile=json.load(f)
        for i in infile["nets"]:
            if "var" in i.keys():
                weights.append(i["var"])
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
