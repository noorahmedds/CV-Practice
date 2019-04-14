//Global variables
PImage im;
PImage depth;
int crudeness = 2;
float parallaxSpeed = 0.1; //set this between 1 and 0. the closer to 1 the faster the parallax
float cameraDepth = 0;

PVector []worldPoints;

//Primitive setup
int primitiveRadius = 3;
boolean redraw = true;

color getColorAt(PImage image, int x, int y){
  return (image.pixels[(y*image.width) + x]);
}

float colorToDepth(PImage _depth, int x, int y){
  color c = getColorAt(_depth, x, y);
  return (brightness(c) + cameraDepth);
}

void setPositionsAt(PVector p, int x, int y){
  worldPoints[(y*im.width) + x] = p;
}

PVector pixelToWorldPoint(int x, int y){
  return worldPoints[(y*im.width) + x];
}

void setupWorldPoints(int offsetx, int offsety){
  worldPoints = new PVector[im.height * im.width];
  
  for (int x = 0; x < im.width; x++){
    for (int y = 0; y < im.height; y++){
      worldPoints[(y*im.width) + x] = new PVector(x - (width/2) + offsetx, y - (height/2) + offsety, 0);
    }
  }
}

void setup(){ 
  
  size(1300, 600, P3D);
  //colorMode(HSB);
  //background(0);
  readImages();
}

void readImages(){
  //im = loadImage("i3.jpeg");
  //depth = loadImage("d3.jpeg");
  
  im = loadImage("i1.png");
  depth = loadImage("d1.jpg");
  
  im.loadPixels();
  depth.loadPixels();
  
  setupWorldPoints(0, 100);
}

void drawPointCloud(){ //Pass it the image, depth map, offset
//This will be called every frame
//first we reconstruct from the arrays that contain our images by displaying each pixel as an ellipse
//then each pixel will shifted by parametrized by the offset(i.e. the mosue position) and the depth it exists at 
  
  //calculate offset
  float offsetx, offsety;
  offsetx = ((float(mouseX)/float(width)) * 2) - 1;
  offsety = ((float(mouseY)/float(height)) * 2) - 1;
  
  color c;
  PVector pos;
  //float z;
  
  //Loop over image, create ellipses with required position
  for (int y = im.height/4; y < (3*im.height /4); y += crudeness){
    for (int x = im.width/4; x < (3*im.width /4); x += crudeness){
      //set color
      c = getColorAt(im, x, y);
      //stroke(c);
      noStroke();
      fill(c);
      
      //here we set the depth of this particular vertex/primitive
      pos = pixelToWorldPoint(x, y);
      //z = colorToDepth(depth, x, y); //currently just returning brightness of the current pixel in the depth map
      
      //update pos based on offset and initial depth. Offset should be between -2 and 2 for example with multiplicative factor of the depth, update positions in the positions array
      pos.z = brightness(depth.pixels[(y*im.width) + x]) + cameraDepth;
      pos.x += offsetx * pos.z * parallaxSpeed;
      pos.y += offsety * pos.z * parallaxSpeed;
      
      setPositionsAt(pos, x, y);
        
      //draw vertex
      translate(0, 0, pos.z);
      rect(pos.x, pos.y, primitiveRadius, primitiveRadius);
      translate(0, 0, -pos.z);
    }
  }
}

void mouseMoved(){
   redraw = true;
}

void keyPressed(){
  if (keyCode == DOWN){
    cameraDepth -= 100;
  } else if (keyCode == UP){
    cameraDepth += 100;
  }
  
}

void draw(){
    
  //camera(width/2,height/2 - 1000, cameraDepth, // eyeX, eyeY, eyeZ
  //       width/2, width/2, 0, // centerX, centerY, centerZ
  //       0.0, 1.0, 0.0); // upX, upY, upZ
  
   if(redraw){
     //lights();
    //noLoop();
    //return;
    background(0);
    translate(width/2, height/2, 0);
    //rotateY(rotVal);
    //rotVal += 0.01;
    
    drawPointCloud(); 
    redraw = false;
  }
}