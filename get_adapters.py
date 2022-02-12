import pathlib
import s3fs

s3 = s3fs.S3FileSystem(anon=False)

bucket_fac = 's3://kadapter/pretrained_models/fac-adapter'
bucket_lin = 's3://kadapter/pretrained_models/lin-adapter'

dest_fac = pathlib.Path('pretrained_models/fac-adapter/')
dest_lin = pathlib.Path('pretrained_models/lin-adapter/')

if not dest_fac.exists():
    dest_fac.mkdir(parents=True)

if not dest_lin.exists():
    dest_lin.mkdir(parents=True)

for file in s3.ls(bucket_fac):
    file_name = file.split('/')[-1]
    s3.download(file, str(dest_fac.absolute()) + '/' + file_name)
for file in s3.ls(bucket_lin):
    file_name = file.split('/')[-1]
    s3.download(file, str(dest_lin.absolute()) + '/' + file_name)
