# Image Interpolation and enhancement
Combines a high resolution panchromatic image(lacks the chromatic data) with a low resolution multispectral (lacks the resolution data) image first using wavelet fusion, then passes it through a Laplacian filter for sharpening and then passes it through a theano neural network


#Steps To Recreate the Project

<ul>
<li> Download the conda distribution, from <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html" >Here</a> and follow the steps to install the conda distribution successfully</li>
<li>After downloading the distribution activate a conda virtual environment using the command create a new virtual environment using the command conda create -n envname python=x.x anaconda </li>
<li> Clone the project inside the folder you want to recreate it</li>
<li> Download the necessary requirements by using the command pip install -r requirements.txt</li>
<li>Run the  _image_fusion_grayscale_panchromatic.py_ script. It will generate grayscale, blurred and fused images of a park</li>
<li>In order to change the image we have added a few more photos in the resources section, change line 60 of above script in order to change the image on which you will conduct this experiment.</li>
<li>Run the enhance.py script using the command python enhance.py --type=photo --model=repair --zoom=1 #_name_of_the_fused_image_#</li>
<li>A png image of the same image will be generated</li>
</ul>