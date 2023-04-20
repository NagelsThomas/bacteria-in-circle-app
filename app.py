from openpiv import (
    pyprocess,
    tools,
    windef,
)

from io import StringIO

from pathlib import Path
from skimage.morphology import (
    binary_dilation,
    white_tophat,
    disk,
)
import matplotlib.pyplot as plt
import numpy as np

from viktor.views import(
    ImageResult,
    ImageView,
    WebResult,
    WebView,
)
from viktor import (
    ViktorController,
    File,
)
from viktor.parametrization import (
    ViktorParametrization,
    NumberField,
    LineBreak,
    Step,
    Text,
)

def readImage(seq):
    imagePath = Path(__file__).parent/f"bacteria-in-droplet-{seq}.tif"
    image = tools.imread(imagePath)
    return image
        
def maskImage(params, seq):
        imageToMask = readImage(seq).copy()
        mask = readImage(0) > params.step3.mask
        mask = np.logical_xor(mask, white_tophat(mask, disk(3)))
        mask = binary_dilation(mask, disk(7))
        imageToMask[~mask] = 0
        return imageToMask

def searchSizeCalc(windowSize):
    if windowSize <= 32:
        searchSize = 32
    else:
        searchSize = windowSize
    return searchSize


class Parametrization(ViktorParametrization):
    
    step1 = Step('Step 1 - Introduction', views=["showGIF"])
    step1.text1 = Text(
"""
# Bacteria in a droplet demonstration app

Welcome to the bacteria in a droplet app! [OpenPIV](https://openpiv.readthedocs.io/en/latest/#) is
 an open source repository for particle image velocimetry applications, in this app, the OpenPIV 
 module is used to analyse the movement of bacteria in a droplet as can be seen in the GIF. 
 
The OpenPIV package has been co-developed by [Prof. Alex Liberzon](https://www.linkedin.com/in/alexliberzon/) 
at the Turbulence Structure Laboratory, at Tel Aviv Unversity and a OpenPIV consortium. 

The package has a large amount of contributors, you can find them in the [openPIV users and community](https://groups.google.com/g/openpiv-users/c/Us_q7h3Uri8/m/1p8XAYkHCQAJ).
If you would like to find out more about the data source, all the imagery can be found [here](https://drive.google.com/drive/folders/1DCt4EbkYoGHun22IWgT5OGWpmQlsMuHs).
This app reproduces the analysis of a bacteria in a droplet developed for the [OpenPIV-Python examples](https://github.com/OpenPIV/openpiv-python-examples/blob/main/notebooks/bacteria_in_a_circle.ipynb)
based on the data presented by [Zhengyhang Liu](https://www.linkedin.com/in/zhengyang-liu-b62986115/). The data were produced by me in the lab of [Eric Clement](https://blog.espci.fr/eclement) at PMMH, ESPCI Paris 

"""
    )

    step2 = Step('Step 2 - Images used for analysis', views=["showImages"])
    step2.text2 = Text(
        """
## Image preperation

In this view, the user can see the two images that are selected for the analysis.
The OpenPIV functions will require two images to compare and determine the direction
and velocity of the particles relative to each other. 

In the view, the images are placed next to each other for you to see the small differences.
In the next step we are going to let the OpenPIV functions calculate the direction and velocity
of the movement of the bacteria. 
        """

    )

    step3 = Step('Step 3 - Perform Analysis', views=["generateResult"])
    step3.text3 = Text(
        """
## PIV Analysis
In this step, the user is able to modify the parameters of the analysis. A detailed description
of the functions used can be found in the [OpenPIV documentation](https://openpiv.readthedocs.io/en/latest/). 
As can be seen, the bacteria do not move random but seem to follow each other. Similar to a flock
of birds or a herd of sheep. Hence PIV is a very useful tool in flow visualisation, can you think of more applications? 

        """
    )

    step3.mask = NumberField(
        "Mask parameter",
        default=150,
        min = 100,
        flex=50,
        description="The Mask value determines the size of the circle of analysis",
    )
    step3.lb1 = LineBreak()
    step3.windowSize = NumberField(
         "Change window size",
         min=32,
         default=32,
         max=48,
         step=8,
         flex=50,
         description="The window size determines the amount of pixels across to take per point",
    )

    step3.lb2 = LineBreak()
    step3.overlap = NumberField(
         "Change overlap",
         min=0,
         default=20,
         step=4,
         flex=50,
         description="This adds points by overlapping the windows thus increasing the amount of arrows",
    )

    finalStep = Step("What's next?", views=["final_step"])

class Controller(ViktorController):
    viktor_enforce_field_constraints = True  # Resolves upgrade instruction https://docs.viktor.ai/sdk/upgrades#U83

    label = 'My Entity Type'
    parametrization = Parametrization

    @ImageView("GIF of droplet", duration_guess=1)
    def showGIF(self, params, **kwargs):
        filePath = Path(__file__).parent/'bacteria.gif'
        return ImageResult(File.from_path(filePath))

    @ImageView("Prepped Images", duration_guess=1)
    def showImages(self, params, **kwargs):
        fig = plt.figure()
        plt.imshow(np.c_[readImage(0),readImage(1)])
        plt.xticks([])
        plt.yticks([])
        plt.title("Image 1 and 2 used for analysis")
        svgData = StringIO()
        fig.savefig(svgData, format='svg')
        plt.close()
        return ImageResult(svgData)
    
    @ImageView("Displacement & Velocity", duration_guess=10)
    def generateResult(self, params, **kwargs):
        maskFirstImage = maskImage(params,0)
        maskSecondImage = maskImage(params,1)
        fig = self.simple_piv(maskFirstImage, maskSecondImage, params)
        svgData = StringIO()
        fig.savefig(svgData, format='svg')
        plt.close()
        return ImageResult(svgData)
    
    @WebView(" ", duration_guess=1)
    def final_step(self, params, **kwargs):
        """Initiates the process of rendering the last step."""
        html_path = Path(__file__).parent / "final_step.html"
        with html_path.open() as f:
            html_string = f.read()
        return WebResult(html=html_string)

    
    @staticmethod
    def simple_piv(im1, im2, params):
        if isinstance(im1, str):
            im1 = tools.imread(im1)
            im2 = tools.imread(im2)
        u, v, s2n = pyprocess.extended_search_area_piv(
            im1.astype(np.int32), im2.astype(np.int32), window_size=params.step3.windowSize,
            overlap=params.step3.overlap, search_area_size=searchSizeCalc(params.step3.windowSize)
        )
        x, y = pyprocess.get_coordinates(image_size=im1.shape,
                                        search_area_size=params.step3.windowSize, overlap=params.step3.overlap)
        valid = s2n > np.percentile(s2n, 5)
        fig = plt.figure()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(im1, cmap=plt.get_cmap("gray"), alpha=0.5, origin="upper")
        ax.quiver(x[valid], y[valid], u[valid], -v[valid], scale=70,
                color='r', width=.005)
        
        return fig

    
