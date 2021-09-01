
import sys

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

def set_Tk_var():
    global progress_bar
    progress_bar = tk.IntVar()
    global selectedButton
    selectedButton = tk.StringVar()

def Aboutus():
    print('DeepBrainSeg1_support.Aboutus')
    sys.stdout.flush()

def AxialScroll(*args):
    print('DeepBrainSeg1_support.AxialScroll')
    sys.stdout.flush()

def CorronalScroll(*args):
    print('DeepBrainSeg1_support.CorronalScroll')
    sys.stdout.flush()

def FlairView():
    print('DeepBrainSeg1_support.FlairView')
    sys.stdout.flush()

def GetRadiomics():
    print('DeepBrainSeg1_support.GetRadiomics')
    sys.stdout.flush()

def Get_Segmentation():
    print('DeepBrainSeg1_support.Get_Segmentation')
    sys.stdout.flush()

def Load_Flair():
    print('DeepBrainSeg1_support.Load_Flair')
    sys.stdout.flush()

def Load_T1():
    print('DeepBrainSeg1_support.Load_T1')
    sys.stdout.flush()

def Load_T1ce():
    print('DeepBrainSeg1_support.Load_T1ce')
    sys.stdout.flush()

def Load_T2():
    print('DeepBrainSeg1_support.Load_T2')
    sys.stdout.flush()

def SagitalScroll(*args):
    print('DeepBrainSeg1_support.SagitalScroll')
    sys.stdout.flush()

def SegmentationOverlay():
    print('DeepBrainSeg1_support.SegmentationOverlay')
    sys.stdout.flush()

def T1View():
    print('DeepBrainSeg1_support.T1View')
    sys.stdout.flush()

def T1ceView():
    print('DeepBrainSeg1_support.T1ceView')
    sys.stdout.flush()

def T2View():
    print('DeepBrainSeg1_support.T2View')
    sys.stdout.flush()

def init(top, gui, *args, **kwargs):
    global w, top_level, root
    w = gui
    top_level = top
    root = top

def destroy_window():
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None

if __name__ == '__main__':
    import DeepBrainSeg
    DeepBrainSeg.vp_start_gui()




