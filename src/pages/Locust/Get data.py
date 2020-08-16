import json
import requests
api_url_base="https://services5.arcgis.com/sjP4Ugu5s0dZWLjd/arcgis/rest/services/Swarms_Public/FeatureServer/0/query?where=1%3D1&outFields=STARTDATE,TmSTARTDAT,FINISHDATE,EXACTDATE,PARTMONTH,LOCNAME,LOCRELIAB,LOCPRESENT,CONFIRMATN,TmFINISHDA,AREAHA&outSR=4326&f=json"
def getdata():
    response=requests.get(api_url_base)
    if response.status_code == 200:
        return json.loads(response.content.decode('utf-8'))
    else:
        return None
account_info = getdata()

if account_info is not None:
    print("Here's your info: ")
    print(type(account_info))
    #for k, v in account_info.items():
        #print(v)
else:
    print('[!] Request Failed')
