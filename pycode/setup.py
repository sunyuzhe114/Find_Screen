# http://193.168.0.177:5000/show_weixinWordCloud/?beginDate=2018-05-23&endDate=2018-05-27&userName=all

import os
libs={"flask","pillow","numpy","opencv-python"}
try:
    for lib in libs:
        os.system("pip install " + lib)
    print("istall success")
except:
    print("istall fail")
