## 4.8 Homework

In this homework, we'll deploy the ride duration model in batch mode. Like in homework 1 and 3, we'll use the FHV data. 

You'll find the starter code in the [homework](homework/) directory.


## Q1. Notebook

We'll start with the same notebook we ended up with in homework 1.

We cleaned it a little bit and kept only the scoring part. Now it's in [homework/starter.ipynb](homework/starter.ipynb).

Run this notebook for the February 2021 FVH data.

What's the mean predicted duration for this dataset?

* 11.19
* 16.19 ✅
* 21.19
* 26.19


## Q2. Preparing the output

Like in the course videos, we want to prepare the dataframe with the output. 

First, let's create an artificial `ride_id` column:

```python
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
```

Next, write the ride id and the predictions to a dataframe with results. 

Save it as parquet:

```python
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)
```

What's the size of the output file?

* 9M
* 19M ✅
* 29M
* 39M
  
Make sure you use the snippet above for saving the file. It should contain only these two columns. For this question, don't change the
dtypes of the columns and use pyarrow, not fastparquet. 

```bash
-rw-r--r--  1 ------  -----    19M Jun 22 11:46 result.parquet
```


## Q3. Creating the scoring script

Now let's turn the notebook into a script. 

Which command you need to execute for that?

```base
jupyter nbconvert --to script score.ipynb
```

## Q4. Virtual environment

Now let's put everything into a virtual environment. We'll use pipenv for that.

Install all the required libraries. Pay attention to the Scikit-Learn version:
check the starter notebook for details. 

After installing the libraries, pipenv creates two files: `Pipfile`
and `Pipfile.lock`. The `Pipfile.lock` file keeps the hashes of the
dependencies we use for the virtual env.

What's the first hash for the Scikit-Learn dependency?
```base
sha256:08ef968f6b72033c16c479c966bf37ccd49b06ea91b765e1cc27afefe723920b
```
```base
"hashes": [
                "sha256:08ef968f6b72033c16c479c966bf37ccd49b06ea91b765e1cc27afefe723920b",
                "sha256:158faf30684c92a78e12da19c73feff9641a928a8024b4fa5ec11d583f3d8a87",
                "sha256:16455ace947d8d9e5391435c2977178d0ff03a261571e67f627c8fee0f9d431a",
                "sha256:245c9b5a67445f6f044411e16a93a554edc1efdcce94d3fc0bc6a4b9ac30b752",
                "sha256:285db0352e635b9e3392b0b426bc48c3b485512d3b4ac3c7a44ec2a2ba061e66",
                "sha256:2f3b453e0b149898577e301d27e098dfe1a36943f7bb0ad704d1e548efc3b448",
                "sha256:46f431ec59dead665e1370314dbebc99ead05e1c0a9df42f22d6a0e00044820f",
                "sha256:55f2f3a8414e14fbee03782f9fe16cca0f141d639d2b1c1a36779fa069e1db57",
                "sha256:5cb33fe1dc6f73dc19e67b264dbb5dde2a0539b986435fdd78ed978c14654830",
                "sha256:75307d9ea39236cad7eea87143155eea24d48f93f3a2f9389c817f7019f00705",
                "sha256:7626a34eabbf370a638f32d1a3ad50526844ba58d63e3ab81ba91e2a7c6d037e",
                "sha256:7a93c1292799620df90348800d5ac06f3794c1316ca247525fa31169f6d25855",
                "sha256:7d6b2475f1c23a698b48515217eb26b45a6598c7b1840ba23b3c5acece658dbb",
                "sha256:80095a1e4b93bd33261ef03b9bc86d6db649f988ea4dbcf7110d0cded8d7213d",
                "sha256:85260fb430b795d806251dd3bb05e6f48cdc777ac31f2bcf2bc8bbed3270a8f5",
                "sha256:9369b030e155f8188743eb4893ac17a27f81d28a884af460870c7c072f114243",
                "sha256:a053a6a527c87c5c4fa7bf1ab2556fa16d8345cf99b6c5a19030a4a7cd8fd2c0",
                "sha256:a90b60048f9ffdd962d2ad2fb16367a87ac34d76e02550968719eb7b5716fd10",
                "sha256:a999c9f02ff9570c783069f1074f06fe7386ec65b84c983db5aeb8144356a355",
                "sha256:b1391d1a6e2268485a63c3073111fe3ba6ec5145fc957481cfd0652be571226d",
                "sha256:b54a62c6e318ddbfa7d22c383466d38d2ee770ebdb5ddb668d56a099f6eaf75f",
                "sha256:b5870959a5484b614f26d31ca4c17524b1b0317522199dc985c3b4256e030767",
                "sha256:bc3744dabc56b50bec73624aeca02e0def06b03cb287de26836e730659c5d29c",
                "sha256:d93d4c28370aea8a7cbf6015e8a669cd5d69f856cc2aa44e7a590fb805bb5583",
                "sha256:d9aac97e57c196206179f674f09bc6bffcd0284e2ba95b7fe0b402ac3f986023",
                "sha256:da3c84694ff693b5b3194d8752ccf935a665b8b5edc33a283122f4273ca3e687",
                "sha256:e174242caecb11e4abf169342641778f68e1bfaba80cd18acd6bc84286b9a534",
                "sha256:eabceab574f471de0b0eb3f2ecf2eee9f10b3106570481d007ed1c84ebf6d6a1",
                "sha256:f14517e174bd7332f1cca2c959e704696a5e0ba246eb8763e6c24876d8710049",
                "sha256:fa38a1b9b38ae1fad2863eff5e0d69608567453fdfc850c992e6e47eb764e846",
                "sha256:ff3fa8ea0e09e38677762afc6e14cad77b5e125b0ea70c9bba1992f02c93b028",
                "sha256:ff746a69ff2ef25f62b36338c615dd15954ddc3ab8e73530237dd73235e76d62"
            ],
```

## Q5. Parametrize the script

Let's now make the script configurable via CLI. We'll create two 
parameters: year and month.

Run the script for March 2021. 

What's the mean predicted duration? 

* 11.29
* 16.29 ✅
* 21.29
* 26.29

Hint: just add a print statement to your script.


## Q6. Docker container 

Finally, we'll package the script in the docker container. 
For that, you'll need to use a base image that we prepared. 

This is how it looks like:

```
FROM python:3.9.7-slim

WORKDIR /app
COPY [ "model2.bin", "model.bin" ]
```

(see [`homework/Dockerfile`](homework/Dockerfile))

We pushed it to [`agrigorev/zoomcamp-model:mlops-3.9.7-slim`](https://hub.docker.com/layers/zoomcamp-model/agrigorev/zoomcamp-model/mlops-3.9.7-slim/images/sha256-7fac33c783cc6018356ce16a4b408f6c977b55a4df52bdb6c4d0215edf83af5d?context=explore),
which you should use as your base image.

That is, this is how your Dockerfile should start:

```docker
FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

# do stuff here
```

This image already has a pickle file with a dictionary vectorizer
and a model. You will need to use them.

Important: don't copy the model to the docker image. You will need
to use the pickle file already in the image. 

Now run the script with docker. What's the mean predicted duration
for April 2021? 


* 9.96 ✅
* 16.55
* 25.96
* 36.55


## Bonus: upload the result to the cloud (Not graded)

Just printing the mean duration inside the docker image 
doesn't seem very practical. Typically, after creating the output 
file, we upload it to the cloud storage.

Modify your code to upload the parquet file to S3/GCS/etc.


## Submit the results

* Submit your results here: https://forms.gle/pFAYjTFqFMJELG819
* It's possible that your answers won't match exactly. If it's the case, select the closest one.
* You can submit your answers multiple times. In this case, the last submission will be used for scoring.

## Deadline

The deadline for submitting is 25 June 2022 (Saturday) 23:00 CEST. After that, the form will be closed.


## Solution

After the deadline, we'll post the solution here


## Publishing the image to dockerhub

This is how we published the image to Docker hub:

```bash
docker build -t mlops-zoomcamp-model:v1 .
docker tag mlops-zoomcamp-model:v1 agrigorev/zoomcamp-model:mlops-3.9.7-slim
docker push agrigorev/zoomcamp-model:mlops-3.9.7-slim
```

