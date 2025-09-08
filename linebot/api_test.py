from toysub import ToysubClient

client = ToysubClient("http://3.236.146.57:5000")  
reply = client.ask("推薦適合一歲的玩具")

print("回答：", reply["text"])
print("圖片：", reply["image"])