# Welcome to whatever this is...

Brainded project I spent my summer working on... here are some photos I guess... I basically tried to clone [this thing](https://github.com/ProbableTrain/MapGenerator). 

### Motivation 
The existing map generator is amazing, however, I have no clue what I am doing, also I want to have the ability to generate different shape of roads (having a crossection of a mesh trace the hyperstreamline) instead of blender extrude. I want it to be in unity, but I have negative programming skills so I decided to re-write the tensor field generator 

### Technical differences
   The Typescript version did not have the Line Intergral Convolution. I added it to my version
  ![Screenshot 2024-10-02 194433](https://github.com/user-attachments/assets/5111e778-30f7-4c10-ab22-7e1dd28fb2bc)

I used floodfill instead of tracing a city block circularly so that LIC can also display local tensor fields.
This picture depicts a local tensor field, which is independent from the parent tensor field. It is colored green to differentiate. The semi circle at the *bottom left corner* is contained within the cityblock. 

  ![Screenshot 2024-10-09 205828](https://github.com/user-attachments/assets/fc56a6bc-a6d2-4c84-b4b7-a0561420ea0d)

  ![Screenshot 2024-10-04 214512](https://github.com/user-attachments/assets/604043e7-c819-4925-af7b-f72072d9e9ce)

    
  
 

