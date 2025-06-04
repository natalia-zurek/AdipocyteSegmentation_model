/* Ver 1.16 - 8th April 2019
Adiposoft---------------------------------------------
is an automated free software for the analysis of white adipose tissue cellularity in histological sections.
It was developed by Miguel Galarraga (mgalarraga@unav.es), and modified by Mikel Ariz (mikelariz@unav.es), of the Imaging Unit at the Center for Applied Medical Research (CIMA) of the University of Navarra, in Pamplona (Spain)
---------------------------------------------------------
The software can be freely used for research purposes, but it can not be distributed without our consent. In addition, we kindly request that you quote the following paper in any publication containing data obtained making use of our software:
1: Boqué N, Campión J, Paternain L, García-Díaz DF, Galarraga M, Portillo MP, Milagro FI, Ortiz de Solórzano C, Martínez JA. Influence of dietary macronutrient composition on adiposity and cellularity of different fat depots in Wistar rats. J Physiol Biochem. 2009 Dec;65(4):387-95. PubMed PMID: 20358352.

/*** License: GPL  ***
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
***/

/*Acknowledgements
 * Jerome Mutterer for its Action Bar Plugin that allows to stick buttons in edition mode * 
 * Fiji and ImageJ developers..
 */

 /*Changelog Version 1.15
  * - Fixes regarding manual edition functions, to save correctly both .xls with results and .jpg analyzed image
  * - Fixes regarding batch processing of a directory of images
*/

var r=0.252, d=10, D=10000, mic="Microns", type="A list of nested Directories - Batch Mode", auto=true, dir="C:/Projects/Ovarian cancer/TCGA", filename="",mode="End",exedges=true;
 
macro "Adiposoft Action Tool - Cf00T4d15A"
{
	
//run("Install...", "install=/home/mgalarraga/SACOS/publico/InterCIMA/MIGUEL/adiposoft/Versions/1.13/Adiposoft.ijm");
//eval("js", "IJ.getInstance().hide()");
run("Colors...", "foreground=black background=white selection=yellow");
//0.3446; %objective of 20x%
//run("Properties...", "channels=1 slices=1 frames=1 unit=um pixel_width=0.3446000 pixel_height=0.3446000 voxel_depth=0.3446000 frame=[0 sec] origin=0,0");
showStatus("Choose parameters");
Dialog.create("-Adiposoft_1.16-")
typesArray=newArray("One Image","A Directory","A list of nested Directories - Batch Mode");
mpArray=newArray("Microns","Pixel");
//Dialog.addMessage(" -- Welcome to Adiposoft -- ")
Dialog.addMessage("Choose parameters")
Dialog.addCheckbox("Auto mode (Manual edition off)",auto);
Dialog.addCheckbox("Exclude on edges",exedges);
//Dialog.addCheckbox("Process a folder",false);
//Dialog.addCheckbox("Output in microns",false);
Dialog.addChoice("Output units",mpArray,mic);
//Dialog.addNumber("Calibration: microns per pixel: ", 0.346) ;
Dialog.addChoice("How many images do you want to analyze?",typesArray,type);

Dialog.addHelp("http://fiji.sc/Adiposoft") 
Dialog.show();

auto=Dialog.getCheckbox();
exedges=Dialog.getCheckbox();
mic=Dialog.getChoice();
type=Dialog.getChoice();

out=1;
if(mic=="Pixel"){
	d=60;
	D=250;
	Dialog.create("-Calibration-")
	Dialog.addNumber("Minimum diameter:", d);
	Dialog.addNumber("Maximum diameter:", D);
	Dialog.show();
	r=1;
	d=Dialog.getNumber();
	D=Dialog.getNumber();
}
else{	
	Dialog.create("-Calibration-")
	Dialog.addNumber("Microns per pixel:", r);
	Dialog.addNumber("Minimum diameter:", d);
	Dialog.addNumber("Maximum diameter:", D);
	Dialog.show();
	r=Dialog.getNumber();
	d=Dialog.getNumber();
	D=Dialog.getNumber();
	
}

if(type=="One Image"){
	if(isOpen(File.name)){}
	else{	
		filename=File.openDialog("Choose an Image for the analysis");
		open(filename);}
	wait(200);
	run("ROI Manager...");
	OutDir=getDirectory("Choose output Directory");
	//save the name and dir of image
	name=getInfo("image.filename");
	dir=getInfo("image.directory");
	//
	run("Select All");
	run("Put Behind [tab]");
	setBatchMode(true);
	roiManager("Add");
	//run("Add to Manager");
	eval("js","RoiManager.getInstance().hide()"); 
	Adiposoft(OutDir,dir,name,auto,r,d,D,type);
	setBatchMode(false);
	if(auto==1){
		
		
	}

}
	
if(type=="A Directory"){
	if(!auto){showMessage("Adiposoft", "Manual edition is not available in folder mode for this version of Adiposoft. Auto is on");
	auto=1;}
	
	InDir=getDirectory("Choose input Directory");
	OutDir=getDirectory("Choose output Directory");//before InDir
	AdiposoftDir(OutDir,InDir,auto,r,d,D,type);}

if(type=="A list of nested Directories - Batch Mode"){
	
	if(!auto){showMessage("Adiposoft", "Manual edition is not available in folder mode for this version of Adiposoft. Auto is on");
	auto=1;}
	InDir=getDirectory("Choose a Directory");
	OutDir=getDirectory("Choose output Directory");//before InDir
	list=getFileList(InDir);
	L=lengthOf(list);
	//L=1;
	//auto=1;
	//type=1;
	D=File.directory;
	print(D);
	//setBatchMode(true);
	for (i=0; i<L; i++)
	{
		//print(OutDir+list[i]);
		if(File.isDirectory(InDir+list[i]))
		{
			//print(OutDir+list[i]);
			dir=InDir+list[i];
			AdiposoftDir(OutDir,dir,auto,r,d,D,type);
		}

	}
	
	//setBatchMode(false);
eval("js", "IJ.getInstance().show()");
//End - show exit window
	if(auto==1){
		//close();
		showMessage("Succesfully processed. Results stored in '" + OutDir+"'");
		run("Adiposoft");
	

}

}

}



function AdiposoftDir (OutDir,InDir,auto,r,d,D,type) {

list=getFileList(InDir);
L=lengthOf(list);
//L=1;
//auto=1;
//type=1;

//get the name of the folder to create it inside the output
name=InDir;//list[i];
strA=split(name,File.separator);
l=lengthOf(strA);
name1=strA[l-1];
//print(name1);
//creates the output directory hierarchy
AuxDir=OutDir+name1+File.separator; //name with results? xlsFiles?
File.makeDirectory(AuxDir);

for (i=0; i<L; i++)
{
	if(endsWith(list[i],".tif")||endsWith(list[i],".jpg")||endsWith(list[i], '.png'))
	{
		fname=InDir+list[i];
		//print(list[i]);
		//run("Select All");
		//run("Add to Manager");
		open(fname);
		wait(200);
		run("ROI Manager...");
		run("Select All");
		run("Put Behind [tab]");
		setBatchMode(true); 
		run("Add to Manager");
		eval("js","RoiManager.getInstance().hide()"); 
		
		//print(AuxDir);
		//waitForUser("aa");
		Adiposoft(AuxDir,InDir,list[i],auto,r,d,D,type);
		roiManager("reset");
		run("Clear Results");
		close();
		setBatchMode(false); 
	}
}
list=getFileList(AuxDir);
//print(list[0]); ////REVISAR***************************************
L=lengthOf(list);
first=1;
//waitForUser("");
//Merge all the .xls files

//check for incompatibilities
setBatchMode(false); 
//
/*
for (i=0; i<L; i++)
{

	if(endsWith(list[i],".xls"))
	{
		fname=AuxDir+list[i];
		open(fname);

		//selectWindow(list[i]);
		IJ.renameResults("Results");
	
		selectWindow("Results");
		if(first==1){
			IJ.renameResults("Total");
			first=0; 	
			//waitForUser("b");
		}
			
		else{	
			nr=nResults;
			Alabel = newArray(nr); 
			Aarea = newArray(nr); 
			Adeq = newArray(nr); 
			
			for(j=0;j<nr;j++)
			{
				//COPY
				//Alabel[j]=getResult("Label",j);
				Aarea[j]=getResult("Area",j);
				Adeq[j]=getResult("D-eq",j);
			}
			selectWindow("Results");
			run("Close");
			wait(100);
			
			selectWindow("Total");
			IJ.renameResults("Results"); 
			nt=nResults;
			for(j=0;j<nr;j++)
			{
			//PASTE
				setResult("Label",nt+j,list[i]);
				setResult("Area",nt+j,Aarea[j]);
				setResult("D-eq",nt+j,Adeq[j]);
			}
			selectWindow("Results");
			IJ.renameResults("Total");
		}
				
	}
}
//save
selectWindow("Total");
IJ.renameResults("Results");
saveAs("Results", OutDir+name1+".xls");
*/

//check for incompatibilities 
setBatchMode(true);


/*
//waitForUser("aa2");
selectWindow("Total");
saveAs("Text", OutDir+name1+".txt");
run("Close");	
File.rename(OutDir+name1+".txt",OutDir+name1+".xls");
*/


//End - show exit window
if(type=="A Directory"){
	if(auto==1){
		//close();
		q=getBoolean("Succesfully processed. Results stored in '" + OutDir+". Do you want to view Results file'");
		if(q){
			open(OutDir+name1+".xls");
			waitForUser("Press ok when finished");
			if(isOpen(name1+".xls")){
				run("Close");
			}
			
		}
		run("Adiposoft");
	}
}	

}

function Adiposoft(output,dir,filename,auto,r,d,D,type) {
	//eval("js", "IJ.getInstance().show()");
	//eval("js","RoiManager.getInstance().hide()"); 
	run("Clear Results");
	roiManager("reset");
	//wait(100);
	eval("js","IJ.getInstance().setAlwaysOnTop(true)");
	//if(auto==0){	eval("js",IJ.run("Action Bar","/plugins/ActionBar/Edition.txt");)}
	if(isOpen(dir+filename)){}
	else{	open(dir+filename);}
	/*print(dir+filename);
	print(filename);
	print(r);
	selectWindow("Log");
	saveAs("Text",output+"imInfo.txt");
	run("Close");	*/
	MyTitle=getTitle();
	showStatus(MyTitle);
	//run("Options...", "iterations=2 count=1 edm=Overwrite do=Nothing");
	//run("Options...", "iterations=2 count=1 edm=Overwrite");
	//setBatchMode(true); 
	showProgress(0.2);
	//run("RGB to Luminance");
	run("8-bit");
	rename("luminance");
	run("Subtract Background...", "rolling=50 light");
	run("Enhance Contrast", "saturated=1 normalize");
	selectWindow("luminance");
	wait(100);
	//run("Threshold...");
	//run("Auto Threshold", "method=Percentile");
	setAutoThreshold("Percentile dark");
	wait(100);
	getThreshold(lower, upper);
	wait(100);
	/*
	selectWindow("Log");
	str=getInfo("window.contents");
	strA=split(str,"\n");
	strB=split(strA[0]," ");
	strC=strB[1];*/
	run("Gaussian Blur...", "sigma=1");
	showProgress(0.3);
	R=parseInt(lower)*0.98;//(89/90); //89 for normal images 91 for low marker.
	//print(R);
	//selectWindow("Log");
	//run("Close"); 
	setThreshold(R, 255);
	run("Convert to Mask");
	//run("Dilate");
	run("Median...", "radius=3");
	showProgress(0.4);
	run("Options...", "iterations=2 count=1 pad edm=Overwrite do=Nothing");
	//run("Options...", "iterations=2 count=1 edm=Overwrite do=Nothing");
	//run("Options...", "iterations=2 count=1 black edm=Overwrite do=Nothing");
	run("Open");
	showProgress(0.5);
	//Filling holes in a more appropiate way
	run("Invert");
	//run("Analyze Particles...", "size=8000-Infinity circularity=0-1 show=Masks");
	//run("Analyze Particles...", "size=30-Infinity pixel show=Masks");
	run("Analyze Particles...", "size=8000-Infinity pixel circularity=0-0.30 show=Masks");
	run("Invert");
	selectWindow("luminance");
	close();
	selectWindow("Mask of luminance");
	rename("luminance");

	//run("Fill Holes");
	showProgress(0.6);
	MyTitle2=getTitle();
	run("Duplicate...", "title=ws");
	run("Distance Map");
	showProgress(0.7);
	run("Find Maxima...", "noise=3 output=[Segmented Particles] light");
	selectWindow("ws");
	close();
	imageCalculator("AND", MyTitle2,"ws Segmented");
	selectWindow("ws Segmented");
	close();
	showProgress(0.8);
	//Measurements
	//run("Fill Holes");
	//run("Set Measurements...", "area standard feret's display redirect=None decimal=3");
	run("Set Measurements...", "area display redirect=None decimal=2");

	//run("Properties...", "channels=1 slices=1 frames=1 unit=um pixel_width=0.3446000 pixel_height=0.3446000 voxel_depth=0.3446000 frame=[0 sec] origin=0,0");
	selectWindow("luminance");
	run("Properties...", "channels=1 slices=1 frames=1 unit=um pixel_width="+r+" pixel_height="+r+" voxel_depth="+r+" frame=[0 sec] origin=0,0");
	//m=280*r*r/(0.35*0.35);
	//M=8000*r*r/(0.35*0.35);
	
		m=PI*pow(d/2,2);//do not touch now.. next versions	
		M=PI*pow(D/2,2);
	//run("Analyze Particles...", "size="+m+"-"+M+" circularity=0.4-1.00 display exclude clear include add");
	if (exedges){
		run("Analyze Particles...", "size="+m+"-"+M+" circularity=0.45-1.00 display exclude clear add");
	}
	else{
		run("Analyze Particles...", "size="+m+"-"+M+" circularity=0.45-1.00 display clear add");
	}
	showProgress(0.95);
	selectWindow("luminance");
	close();
	//run("Add Image...", MyTitle + "x=0 y=0 opacity=100");

	//if(auto==1){}
	//else{
	//setBatchMode(false); }
	//setBatchMode(false);
	//Add the equivalent diameter to results

	if(nResults==0){//check if no adypocites detected, then inform and close - continue if directory
		//waitForUser("No adypocites detected! Please, check parameters and input image. Image will not be analyzed");
		showMessage("Warning!", "No adypocites detected! Please, check parameters and input image.\nImage will not be analyzed");
		close;
		if(type=="One Image"){
			//run again
			run("Adiposoft");
		}	
	}
	else{
		selectWindow("Results");
	
		
		
		for(i=0;i<nResults;i++){
			a=getResult("Area", i);
			d=sqrt(4*a/3.1416);
			setResult("D-eq",i,d);	
			setResult("Label",i,MyTitle);
		}
		//waitForUser("a");
		//saveAs("Results", output+MyTitle+".xls");
	
		//save roi
		roiManager("Select All");
		//roiManager("Save", output+"infoRoi.zip");
		
		//save roi as a label mask
		newImage("Labeling", "16-bit black", getWidth(), getHeight(), 1);

		for (index = 0; index < roiManager("count"); index++) {
			roiManager("select", index);
			setColor(index+1);
			fill();
		}

		resetMinAndMax();
		run("glasbey");
		selectWindow("Labeling");
		saveAs("tif",output+MyTitle);
		
		//save image
		selectWindow(MyTitle);	
		roiManager("Show All with labels");
		roiManager("Show All");
		//if(auto==1){
			//run("Flatten");
			run("Duplicate...", "d");
			if(auto){
			run("From ROI Manager");}
			run("Flatten");
			wait(200);
			//saveAs("Jpeg",output+"Analyzed"+MyTitle);
			wait(200);
			close();
		//	}
		//else{
			//save after closing in editor mode
			//saveAs("Jpeg",output+"Analyzed"+MyTitle);
		//}
		
	
		//eval("js", "IJ.getInstance().hide()");
		//eval("js", "IJ.getInstance().show()");

		setBatchMode("show");

		eval("js","IJ.getInstance().setAlwaysOnTop(false)");

		
		selectWindow(MyTitle);
			close();
			
		if(auto==1){

			if(type=="One Image"){
			
			// MIKEL: AQUÍ METO MANO A VER SI LO ARREGLO
			
			//eval("js","RoiManager.getRoiManager()"); // ESTO PARECE QUE TAMPOCO FUNCIONA...
			eval("js","RoiManager.getInstance().show()"); 
			
			//selectWindow("ROI Manager");			MAG
			//run("Close");}	MAG
			
			//roiManager("reset");
			setBatchMode(false);
			//if(type=="One Image"){
			wait(100);
			//selectWindow("Results");
			//setBatchMode(true); MAG
	
			//SHOW updated results
			if(isOpen("Results"))
			{
			selectWindow("Results");
			run("Close");
			}
			
			//get info
			//output=getInfo("image.directory");
			dir=getInfo("image.directory");
			
			//Fstring=File.openAsString(dir+"imInfo.txt");
			//strA=split(Fstring,"\n");
			MyTitle=filename;//strA[1];
			open(output+MyTitle+".xls");
			
			//Open the toolbar
			showMessage("Adiposoft 1.15","Succesfully processed. Results stored in '" + OutDir+"'");
			close;
			selectWindow("Results");
			run("Close");
			run("Adiposoft");
			
			//run("Action Bar","/plugins/ActionBar/Edition.txt");
			//run("Action Bar","/plugins/ActionBar/Edition.txt");
			}
		
		}
		else{
			selectWindow("Results");
			run("Close");
			setBatchMode(false);
			//selectWindow(MyTitle);
			roiManager("Show All with labels");
			roiManager("Show All");
			getLocationAndSize(x, y, width, height);
			setLocation(x+1,y);
			setBatchMode(true);
			//close();
			//open(dir+MyTitle);
			//selectWindow(MyTitle);
			showStatus("Edition Mode");
			setTool("freehand");
			//run("Action Bar","/plugins/ActionBar/Edition.txt");
			Edition();
			while(mode!="End"){
				Edition();
			}
			open(output+MyTitle+".xls");
			//Open the toolbar
			showMessage("Adiposoft 1.15","Succesfully processed. Results stored in '" + OutDir+"'");
			close;
			selectWindow("Results");
			run("Close");
			run("Adiposoft");
			
				
			
		
			//exit();
		}
	
		//eval("js","RoiManager.getInstance().show()"); 
	}
}

//FUNCTIONS FOR MANUAL MODE -- EDITION
function Edition(){
	//offer manual edition menu
	Dialog.create("-Adiposoft_1.15- Manual Edition")
	typesArray=newArray("End","Delete","Add","Merge","Separate","Undo");
	Dialog.addMessage("Choose Mode for manual edition")
	//Dialog.addChoice("Edition mode",typesArray,mode);
	Dialog.addChoice("Edition mode",typesArray,typesArray[0]);
	Dialog.addHelp("http://fiji.sc/Adiposoft/manual") 
	Dialog.show();
	
	mode=Dialog.getChoice();
	if(mode=="Delete"){
		delete();
		
	}
	if(mode=="Add"){
		add();
		
	}
	if(mode=="Merge"){
		merge();
		
	}
	if(mode=="Separate"){
		separate();
		
	}
	if(mode=="Undo"){
		undo();
		
	}
	if(mode=="End"){
		end();
		
	}
}


function delete(){
	//save Roi
		dir=getInfo("image.directory");
		n=roiManager("count");		
	
		roiManager("Select All");
		roiManager("Save", dir+"infoRoi.zip");
		//
		run("Select None");
		//free selection
				
		//setTool("hand");
		setTool("oval");

		waitForUser("Please select a cell to delete and press ok when ready");
		//check if we have a selection
		getDimensions(width, height, channels, slices, frames);
		getSelectionBounds(x, y, w, h);
		if(w==width){
			showMessage("Edition", "You should mark a cell to delete. Nothing will be deleted.");
			}
		else{
		//start now
		roiManager("Delete");
		//get info
		//output=getInfo("image.directory");
		dir=getInfo("image.directory");
		//Fstring=File.openAsString(dir+"imInfo.txt");
		//strA=split(Fstring,"\n");
		MyTitle=filename;//strA[1];
		
		//r=strA[2];
		//close();
		//multimeasure
		run("Set Measurements...", "area display redirect=None decimal=2");
		//run("Properties...", "channels=1 slices=1 frames=1 unit=um pixel_width=0.3446000 pixel_height=0.3446000 voxel_depth=0.3446000 frame=[0 sec] origin=0,0");
		run("Properties...", "channels=1 slices=1 frames=1 unit=um pixel_width="+r+" pixel_height="+r+" voxel_depth="+r+" frame=[0 sec] origin=0,0");
		n=roiManager("count");
		/*R=newArray(n);
		for (i=0;i<n;i++)
		{
			R[i]=i;
		}
		roiManager("select", R);*/
		roiManager("Select All");
		roiManager("Measure");
		//save results
		selectWindow("Results");
			
			for(i=0;i<nResults;i++){
				a=getResult("Area", i);
				d1=sqrt(4*a/3.1416);
				setResult("D-eq",i,d1);	
				setResult("Label",i,MyTitle);
			}			
			
			saveAs("Results", output+MyTitle+".xls");
			run("Close");
		//save image
		//selectWindow(MyTitle);	
		roiManager("Show All with labels");
		roiManager("Show All");
		run("Flatten");
		wait(200);
		saveAs("Jpeg",output+"Analyzed"+MyTitle);
		close();
		}
}
function add(){

	//save Roi
		dir=getInfo("image.directory");
		n=roiManager("count");
		
		roiManager("Select All");
		roiManager("Save", dir+"infoRoi.zip");

		run("Select None");
		setTool("freehand");
		waitForUser("Please draw a cell to add and press ok when ready");
		//check if we have a selection
		getDimensions(width, height, channels, slices, frames);
		getSelectionBounds(x, y, w, h);
		if(w==width){
			showMessage("Edition", "Yo should draw a region to add. Nothing will be added.");
			exit();}
		//start now
		run("Add to Manager");
		//get info
		//output=getInfo("image.directory");
		dir=getInfo("image.directory");
		//Fstring=File.openAsString(dir+"imInfo.txt");
		//strA=split(Fstring,"\n");
		MyTitle=filename;//strA[1];
		//r=strA[2];
		//close();
		//multimeasure
		run("Set Measurements...", "area display redirect=None decimal=2");
		//run("Properties...", "channels=1 slices=1 frames=1 unit=um pixel_width=0.3446000 pixel_height=0.3446000 voxel_depth=0.3446000 frame=[0 sec] origin=0,0");
		run("Properties...", "channels=1 slices=1 frames=1 unit=um pixel_width="+r+" pixel_height="+r+" voxel_depth="+r+" frame=[0 sec] origin=0,0");
		n=roiManager("count");
		/*R=newArray(n);
		for (i=0;i<n;i++)
		{
			R[i]=i;
		}
		roiManager("select", R);*/
		roiManager("Select All");
		roiManager("Measure");
		//save results
		selectWindow("Results");
			
			for(i=0;i<nResults;i++){
				a=getResult("Area", i);
				d2=sqrt(4*a/3.1416);
				setResult("D-eq",i,d2);	
				setResult("Label",i,MyTitle);
			}
			
			saveAs("Results", output+MyTitle+".xls");
			run("Close");
		//save image
		//selectWindow(MyTitle);	
		roiManager("Show All with labels");
		roiManager("Show All");
		run("Flatten");
		wait(200);
		saveAs("Jpeg",output+"Analyzed"+MyTitle);
		close();
	
}
function merge(){
	//save Roi
		dir=getInfo("image.directory");
		n=roiManager("count");
		MyTitle=filename;
		roiManager("Select All");
		roiManager("Save", dir+"infoRoi.zip");
		//
		//r=0.3466;
		dir=getInfo("image.directory");
		setTool("freehand");
		waitForUser("Please draw an empty area to merge cells and press ok when ready"); 
		
		//check if we have a selection
		getDimensions(width, height, channels, slices, frames);
		getSelectionBounds(x, y, w, h);
		if(w==width){
			showMessage("Edition", "Yo should draw a region to merge two cells. Nothing will be merged.");
			exit();}
		//start now
		//setBatchMode(true); 
		run("Add to Manager");
		//output=getInfo("image.directory");

		
		//rename("luminance");
		//edit image
		//make a selection to merge 2 adypocites
		
		run("Remove Overlay");
		
		run("RGB to Luminance");
		run("Select All");
		run("Clear", "slice");
		rename("luminance");
		
		n=roiManager("count");
		
		roiManager("Select All");
		
		if (n>1){	
		roiManager("Combine");}
		run("Create Mask");
		selectWindow("luminance");
		close();selectWindow("Mask");
		rename("luminance");
		//roimanager reset
		roiManager("reset");
		
		
		
		run("Select All");
		//analyze particles again
		//run("Properties...", "channels=1 slices=1 frames=1 unit=um pixel_width="+r+" pixel_height="+r+" voxel_depth="+r+" frame=[0 sec] origin=0,0");
		run("Properties...", "channels=1 slices=1 frames=1 unit=um pixel_width="+r+" pixel_height="+r+" voxel_depth="+r+" frame=[0 sec] origin=0,0");
		run("Analyze Particles...", "size=0-Inf circularity=0.0-1.00 display clear add");

		
		//add overlay again and save
		selectWindow("luminance");close();
		//run("Add Image...", "orig x=0 y=0 opacity=100");
		//run("Add Image...", "image="+MyTitle+" x=0 y=0 opacity=100");
		//roiManager("Show All with labels");
		//roiManager("Show All");
		//saveAs("Jpeg",output+"Analyzed"+MyTitle);
		//save results
		//Add the equivalent diameter to results
		selectWindow("Results");
		
		for(i=0;i<nResults;i++){
			a=getResult("Area", i);
			d3=sqrt(4*a/3.1416);
			setResult("D-eq",i,d3);	
			setResult("Label",i,MyTitle);
		}
		
		saveAs("Results", output+MyTitle+".xls");
		run("Close");
	
		
		//save image
		//selectWindow(MyTitle);	
		roiManager("Show All with labels");
		roiManager("Show All");
		run("Flatten");
		wait(200);
		saveAs("Jpeg",output+"Analyzed"+MyTitle);
		close();
}
function separate(){
		//save Roi
		dir=getInfo("image.directory");
		n=roiManager("count");
		/*R=newArray(n);
		for (i=0;i<n;i++)
		{
			R[i]=i;
		}
		roiManager("select", R);*/
		roiManager("Select All");
		roiManager("Save", dir+"infoRoi.zip");
		//
		//r=0.3466;
		dir=getInfo("image.directory");
		setTool("freeline");
		setTool("freeline");
		waitForUser("Please draw a line to break a cell and press ok when ready"); 
		
		//check if we have a selection
		getDimensions(width, height, channels, slices, frames);
		getSelectionBounds(x, y, w, h);
		if(w==width){
			showMessage("Edition", "Yo should draw a line to break a cell. Nothing will be done.");
			exit();}
		//start now
		run("Add to Manager");
		//output=getInfo("image.directory");

		
		run("Remove Overlay");
		
		run("RGB to Luminance");
		run("Select All");
		run("Clear", "slice");
		rename("luminance");
		
		n=roiManager("count");
		
		roiManager("Select All");
		
		if (n>1){	
		roiManager("Combine");}
		run("Create Mask");
		selectWindow("luminance");
		close();
		selectWindow("Mask");
		rename("luminance");

		//select the line and use it to separate cell
		roiManager("select", n-1);
		run("Line Width...", "line=2");
		run("Clear", "slice");
		
		//roimanager reset
		roiManager("reset");
		
		run("Select All");

		//analyze particles again
		//run("Properties...", "channels=1 slices=1 frames=1 unit=um pixel_width="+r+" pixel_height="+r+" voxel_depth="+r+" frame=[0 sec] origin=0,0");
		run("Properties...", "channels=1 slices=1 frames=1 unit=um pixel_width="+r+" pixel_height="+r+" voxel_depth="+r+" frame=[0 sec] origin=0,0");
		//run("Analyze Particles...", "size=0-Inf circularity=0.0-1.00 display exclude clear include add");
		run("Analyze Particles...", "size=0-Inf circularity=0.0-1.00 display clear add");
		
		MyTitle=filename;//getTitle();
			
		//add overlay again and save
		selectWindow("luminance");close();
		//run("Add Image...", "orig x=0 y=0 opacity=100");
		//run("Add Image...", "image="+MyTitle+" x=0 y=0 opacity=100");
		//roiManager("Show All with labels");
		//roiManager("Show All");
		//saveAs("Jpeg",output+"Analyzed"+MyTitle);
		//save results
		//Add the equivalent diameter to results
		selectWindow("Results");
		
		for(i=0;i<nResults;i++){
			a=getResult("Area", i);
			d4=sqrt(4*a/3.1416);
			setResult("D-eq",i,d4);	
			setResult("Label",i,MyTitle);
		}
		
		saveAs("Results", output+MyTitle+".xls");
		run("Close");
			
		//save image
		//selectWindow(MyTitle);	
		roiManager("Show All with labels");
		roiManager("Show All");
		run("Flatten");
		wait(200);
		saveAs("Jpeg",output+"Analyzed"+MyTitle);
		close();
	
}
function undo(){
	//undo by restoring infoRoi
	dir=getInfo("image.directory");
	roiManager("reset");
	roiManager("Open", dir+"infoRoi.zip");
	roiManager("Show All with labels");
	roiManager("Show All");

	//get info
		//output=getInfo("image.directory");
		dir=getInfo("image.directory");
	
		MyTitle=filename;//strA[1];

		//close();
		//multimeasure
		run("Set Measurements...", "area display redirect=None decimal=2");
		//run("Properties...", "channels=1 slices=1 frames=1 unit=um pixel_width=0.3446000 pixel_height=0.3446000 voxel_depth=0.3446000 frame=[0 sec] origin=0,0");
		run("Properties...", "channels=1 slices=1 frames=1 unit=um pixel_width="+r+" pixel_height="+r+" voxel_depth="+r+" frame=[0 sec] origin=0,0");
		n=roiManager("count");
		
		roiManager("Select All");
		roiManager("Measure");
		//save results
		selectWindow("Results");
			
			for(i=0;i<nResults;i++){
				a=getResult("Area", i);
				d5=sqrt(4*a/3.1416);
				setResult("D-eq",i,d5);	
				setResult("Label",i,MyTitle);
			}
			
			saveAs("Results", output+MyTitle+".xls");
			run("Close");
		//save image
		//selectWindow(MyTitle);	
		roiManager("Show All with labels");
		roiManager("Show All");
		run("Flatten");
		wait(200);
		saveAs("Jpeg",output+"Analyzed"+MyTitle);
		close();
}
function end(){
	
}
