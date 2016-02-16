import io, re, time
import tornado.httpserver, tornado.ioloop, tornado.options, tornado.web, os.path, random, string
from tornado.options import define, options
import qpcr
import numpy as np
import pandas as pd


analysis_template="""<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN">
<html lang="en">
<head>
  <title>qPCR Analysis - AMALab webtools</title>
    <link rel="stylesheet" type="text/css" media="all" href="/css/amawebtools.css" />
</head>
<body>
  <div class="container">
      <div class="content">
        <div class="page-header">
          <div class="row">
            <p><h1>qPCR data merger</h1></p>
          </div>
        </div>
        <h2>qPCR analysis results</h2>
          <table>
          %s
          </table>
      </div>
  </div>
</body>
</html>
"""
merge_template="""<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN">
<html lang="en">
<head>
  <title>qPCR Analysis - AMALab webtools</title>
    <link rel="stylesheet" type="text/css" media="all" href="/css/amawebtools.css" />
</head>
<body>
  <div class="container">
      <div class="content">
        <div class="page-header">
          <div class="row">
            <p><h1>qPCR data merger</h1></p>
          </div>
        </div>
        <h2>Merge file</h2>
          <table>
            <tr><td><a href="%s">CSV file</a></td></tr>    
          </table>
      </div>
  </div>
</body>
</html>
"""


define("port", default=8080, help="run on the given port", type=int)

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", IndexHandler),
            #(r"/(.*)",tornado.web.StaticFileHandler, {"path": "./"},),
            (r"/upload", UploadHandler),
            (r"/merge", MergeHandler),
            (r"/results/(.*)",tornado.web.StaticFileHandler, {"path": "./results"},),
            (r"/css/(.*)",tornado.web.StaticFileHandler, {"path": "./css"},),
        ]
        tornado.web.Application.__init__(self, handlers)
        
class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("form.html")
        
class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        experimentname=self.get_argument('experiment')
        username=self.get_argument('user')
        html_table=""
        for k,uploadedfile in enumerate(self.request.files['file']):
            original_fname = uploadedfile['filename']
            extension = os.path.splitext(original_fname)[1]
            fname = ''.join(random.choice(string.ascii_lowercase + string.digits) for x in range(6))
            prefix=os.path.splitext(original_fname)[0].replace(' ','_').replace('.','_').replace(',','_')
            up_filename= '%03d'%k+prefix+fname+extension
            up_file = open("uploads/" + up_filename, 'wb')
            up_file.write(uploadedfile['body'])
            up_file.close()
            #self.finish("Analysing qPCR data... it might take some minutes")
            q=qpcr.qpcrAnalysis("uploads/" + up_filename,expname=experimentname,username=username,originalfname=original_fname)
            q.fit()
            q.save_figs()
            q.to_html()
            html_table+="""<tr><td>%s</td>
                             <td><a href=\"%s\">[Report]</a></td>
                             <td><a href=\"%s\">[CSV file]</a></td></tr>\n"""%(original_fname,q.htmlfname,q.fullcsvname)
            del(q)
        self.finish(analysis_template%html_table)
            #self.render("%s.html"%q.info['ID'])
            #self.finish("file" + final_filename + "corresponding to the experiment" +experimentname+ "has been uploaded by"+username)

class MergeHandler(tornado.web.RequestHandler):
    def post(self):
        # Read and all files into a Pandas dataframe:
        aux=[]
        for uploadedfile in self.request.files['file']:
            d=pd.read_csv(io.BytesIO(uploadedfile['body']))
            d["Filename"]=uploadedfile['filename']
            aux.append(d)
        d=pd.concat(aux)
        d.index=range(d.shape[0])
        # Edit data so we have a sample identifier (equal for all repeats)
        d["Sample"]=[re.sub(' +',' ',s.split(': ')[1].strip()) for s in d["Unnamed: 0"]]
        d=d.ix[:,["Sample","Initial Concentration","Filename"]]
        # Compute the maximum number of repeats to be stored 
        dsize=d.ix[:,["Sample"]].groupby("Sample", as_index=False).count()
        maxreps=dsize["Sample"].max() 
        data=pd.DataFrame(data=dsize.index, columns=["Sample"])
        data["Num Repeats"]=np.array(dsize["Sample"])
        for i in range(maxreps):
            data["Rep %d IC"%(i+1)]=np.nan
        data["Max. to Min."]=np.nan
        for i in range(maxreps):
            data["Rep %d Filename"%(i+1)]=""
        # Fill repeats in a sorted manner:
        for i in range(data.shape[0]):
            sample=data.ix[i,"Sample"]
            idx=np.where(d["Sample"]==sample)[0]
            y=np.array(d.ix[idx,"Initial Concentration"])
            isrt=np.argsort(y)
            x=y[isrt]
            for j in range(x.size):
                data.ix[i,"Rep %d IC"%(j+1)]=x[j]
                data.ix[i,"Rep %d Filename"%(j+1)]=d.ix[idx[isrt[j]],"Filename"]
            data.ix[i,"Max. to Min."]=x.max()/x.min()
        mergeID=('%f'%time.time()).replace('.','')
        fnout='results/Merge_%s.csv'%mergeID
        data.to_csv(fnout,index=False)
        self.finish(merge_template%fnout)

        
def main():
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()
