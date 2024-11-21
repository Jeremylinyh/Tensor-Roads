using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UIElements;


public class Matrix : MonoBehaviour
{
    // Start is called before the first frame update
    //public RawImage visualization;
    //public ComputeShader generation;

    private const float shaderdimensions = 8;
    private const int textureSize = 16 * 64;
    private const int scale = 1;
    private const float minimaldistance = 16 * 3;
    private const float StartWider = minimaldistance + 16*1;
    private const float minormindis = minimaldistance / 8;
    private const float majormindis = minimaldistance / 8;
    const float spacing = 3600.0f;

    private static Texture2D DoNotStrip;

    private void Awake()
    {
        DoNotStrip = new Texture2D(0, 0);
    }

    public ComputeShader heightmapComputeShader;
    //public ComputeShader Derive;
    //public RenderTexture heightmapTexture;
    //public RawImage texture;
    private static Dictionary<int2, List<float2>> exisitng = new Dictionary<int2, List<float2>>();
    private static Dictionary<int2, List<float2>> Exisitng = new Dictionary<int2, List<float2>>();
    private static List<float4> ends = new List<float4>();
    private static List<Vector4> intersections = new List<Vector4>();
    private static Dictionary<int2, List<float2>> supplements = new Dictionary<int2, List<float2>>();
    private static bool Chalted = true;
    private static Texture2D Derivitive;

    Color roadcol = new Color(1, 0, 0);
    private static bool majorcolor = true;
    float[] data = new float[textureSize * scale * textureSize * scale * (int)(shaderdimensions * shaderdimensions)];
    void modifyspecified(ref Dictionary<int2, List<float2>> inuse, List<float2> tobeadd)
    {
        foreach (float2 item in tobeadd)
        {
            if (!inuse.ContainsKey((int2)(item / minimaldistance)))
            {
                inuse[(int2)(item / minimaldistance)] = new List<float2>();
            }
            inuse[(int2)(item / minimaldistance)].Add(item);
        }
    }
    void removespecified(ref Dictionary<int2, List<float2>> inuse, List<float2> tobeadd)
    {
        foreach (float2 item in tobeadd)
        {
            if (inuse.ContainsKey((int2)(item / minimaldistance)))
            {
                inuse[(int2)(item / minimaldistance)].Remove(item); //= new List<float2>();
            }
            //inuse[(int2)(item / minimaldistance)].Add(item);
        }
    }
    float2 Trig(float theta, float major, float Co_major)
    {
        float2 result;
        if (theta == float.NaN && false)
        {
            result = new float2();
        }
        else
        {
            result = new float2(Mathf.Cos(theta + major * Mathf.PI / 2), Mathf.Sin(theta + major * Mathf.PI / 2));

        }
        return result;
    }
    float offsettheta(float2 currentposition, float2 Kfunc, float major, float Co_major)
    {
        return readtheta(currentposition, new float2(major / 2 + Co_major * Kfunc.x / 2, Co_major / 2 + major * Kfunc.y / 2));
    }
    float2 rungekutta(float2 currentposition, float major)
    {
        //major = 0;
        currentposition = currentposition + playerposition * (int)textureSize;
        //Debug.Log(playerposition * (int)reallifescale * (int)reallifescale);

        float Co_major = -(major - 1);
        //float Co_major = major;
        //major = -(major - 1);

        float theta = readtheta(currentposition);
        if (theta == float.NaN) { return new float2(); }
        float2 K1 = Trig(theta, major, Co_major);            // new float2(Mathf.Cos(theta + major * Mathf.PI / 2), Mathf.Sin(theta + major * Mathf.PI / 2));

        theta = offsettheta(currentposition, K1, major, Co_major); //readtheta(currentposition + new float2(major/2 + Co_major * K1.y/2, Co_major/2 + major*K1.y/2));
        if (theta == float.NaN) { return new float2(); }
        float2 K2 = Trig(theta, major, Co_major);            //new float2(Mathf.Cos(theta + major * Mathf.PI / 2), Mathf.Sin(theta + major * Mathf.PI / 2));

        theta = offsettheta(currentposition, K2, major, Co_major); // readtheta(currentposition + new float2(major / 2 + Co_major * K2.y/2, Co_major / 2 + major * K2.y / 2));
        if (theta == float.NaN) { return new float2(); }
        float2 K3 = Trig(theta, major, Co_major);           //new float2(Mathf.Cos(theta + major * Mathf.PI / 2), Mathf.Sin(theta + major * Mathf.PI / 2));

        theta = offsettheta(currentposition, K3, major, Co_major); // readtheta(currentposition + new float2(major / 2 + Co_major * K3.y, Co_major / 2 + major * K3.y));
        if (theta == float.NaN) { return new float2(); }
        float2 K4 = Trig(theta, major, Co_major);           //new float2(Mathf.Cos(theta + major * Mathf.PI / 2), Mathf.Sin(theta + major * Mathf.PI / 2));

        return (K1 + 2 * K2 + 2 * K3 + K4) / 6;                                          //new float2(Mathf.Cos(theta + major * Mathf.PI / 2), Mathf.Sin(theta + major * Mathf.PI / 2)); 
    }
    void finishoff(float2 currentposition, Texture2D texture, int major, int sign, out List<float2> tobeadd, Color colorused)
    {
        bool isvalid = true;
        const bool isparallelvalid = true;
        tobeadd = new List<float2>();
        const int Dlookahead = (int)minimaldistance * 2;
        int went = 0;
        //bool isint = false;
        while (isvalid && isparallelvalid && went <= Dlookahead && !(Isitlooped(currentposition, tobeadd)))
        {
            went += 1;
            //float theta = readtheta(currentposition);
            float2 MinorEigenVector = rungekutta(currentposition, major); // new float2(Mathf.Cos(theta + major * Mathf.PI / 2), Mathf.Sin(theta + major * Mathf.PI / 2));
            tobeadd.Add(currentposition);
            currentposition -= MinorEigenVector * sign;

            if (major == 0)
            {
                isvalid = ispointvalid(currentposition, major, ref exisitng, 1f);
                //isint = ispointvalid(currentposition, major, ref exisitng,1f);
            }
            else
            {
                isvalid = ispointvalid(currentposition, major, ref Exisitng, 1f);
                //isint = ispointvalid(currentposition, major, ref Exisitng,1f);
            } //Debug.Log(currentposition);
            //texture.SetPixel((int)currentposition.x, (int)currentposition.y, new Color(1,0,0,1));
            //isvalid = ispointvalid(currentposition, major, ref Exisitng, 1f) && ispointvalid(currentposition, major, ref exisitng, 1f);
        }
        //Debug.Log(went<Dlookahead);
        if (!isvalid)
        {
            foreach (float2 item in tobeadd)
            {
                texture.SetPixel((int)item.x, (int)item.y, colorused);
            }

            texture.Apply();
            //Graphics.Blit(texture, heightmapTexture);
        }
        else
        {
            tobeadd.Clear();
        }
        //Debug.Log("");
    }
    private const float Seed = 4.37585453123f;
    public static Vector2 Random2(Vector2 st)
    {
        st = new Vector2(
            Vector2.Dot(st, new Vector2(127.1f, 311.7f)),
            Vector2.Dot(st, new Vector2(269.5f, 183.3f))
        );
        return -Vector2.one + 2.0f * new Vector2(
            Mathf.Repeat(Mathf.Sin(st.x) * Seed * 100000.0f, 1.0f),
            Mathf.Repeat(Mathf.Sin(st.y) * Seed * 100000.0f, 1.0f)
        );
    }

    public static float Pnoise(Vector2 st)
    {
        Vector2 i = new Vector2(Mathf.Floor(st.x), Mathf.Floor(st.y));
        Vector2 f = st - i;

        Vector2 u = new Vector2(
    f.x * f.x * (3.0f - 2.0f * f.x),
    f.y * f.y * (3.0f - 2.0f * f.y)
);

        Vector2 OO = Random2(i + new Vector2(0.0f, 0.0f));
        Vector2 IO = Random2(i + new Vector2(1.0f, 0.0f));
        Vector2 OI = Random2(i + new Vector2(0.0f, 1.0f));
        Vector2 II = Random2(i + new Vector2(1.0f, 1.0f));

        float dot00 = Vector2.Dot(OO, f - new Vector2(0.0f, 0.0f));
        float dot10 = Vector2.Dot(IO, f - new Vector2(1.0f, 0.0f));
        float dot01 = Vector2.Dot(OI, f - new Vector2(0.0f, 1.0f));
        float dot11 = Vector2.Dot(II, f - new Vector2(1.0f, 1.0f));

        /*Vector2 gradX0 = Vector2.Lerp (OO, IO, u.x);
        Vector2 gradX1 = Vector2.Lerp(OI, II, u.x);
        Vector2 gradY0 = Vector2.Lerp(OO, OI, u.y);
        Vector2 gradY1 = Vector2.Lerp(IO, II, u.y);

        // Calculate final grade
        Vector2 grade = new Vector2(gradX1.x - gradX0.x, gradY1.y - gradY0.y);*/
        float result = Mathf.Lerp(Mathf.Lerp(dot00, dot10, u.x), Mathf.Lerp(dot01, dot11, u.x), u.y);
        return result;
    }
    Vector3 gradientfromnormal(Vector3 n)
    {
        Vector3 z = Vector3.up; // This is the unit vector (0, 1, 0)

        // Compute the cross product n × z
        Vector3 nCrossZ = Vector3.Cross(n, z);

        // Compute the cross product (n × z) × n
        return Vector3.Cross(nCrossZ, n);
    }
    float2 flooroperation(float2 input)
    {
        return new float2(Mathf.Floor(input.x), Mathf.Floor(input.y));
    }
    float2 float2subtraction(float2 input, float value)
    {
        return new float2(input.x - value, input.y - value);
    }
    float3 noised(in float2 p)
    {
        float2 i = flooroperation(p);
        float2 f = p - i;

        float2 u = f * f * f * (f * (float2subtraction(f * 6.0f, 15.0f)) + 10.0f);
        float2 du = 30.0f * f * f * (f * (float2subtraction(f, 2.0f)) + 1.0f);

        float2 ga = Random2(i + new float2(0.0f, 0.0f));
        float2 gb = Random2(i + new float2(1.0f, 0.0f));
        float2 gc = Random2(i + new float2(0.0f, 1.0f));
        float2 gd = Random2(i + new float2(1.0f, 1.0f));

        float va = Vector2.Dot(ga, f - new float2(0.0f, 0.0f));
        float vb = Vector2.Dot(gb, f - new float2(1.0f, 0.0f));
        float vc = Vector2.Dot(gc, f - new float2(0.0f, 1.0f));
        float vd = Vector2.Dot(gd, f - new float2(1.0f, 1.0f));

        return new float3(va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd), // value
                     ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd) + // derivatives
                     du * (u.yx * (va - vb - vc + vd) + new float2(vb, vc) - va));
    }

    float readtheta(float2 currentposition, float2 offset = new float2())
    {
        //if (!majorcolor)
        //{
        //    return 0;
        //}
        //MeshRenderer renderer = renderQuad.GetComponent<MeshRenderer>();
        //Texture2D texture = (Texture2D)renderer.material.mainTexture;
        //float OO = texture.GetPixel((int)currentposition.x,(int)currentposition.y).a;
        //float OI = texture.GetPixel((int)currentposition.x, (int)currentposition.y + 1).a;
        //float II = texture.GetPixel((int)currentposition.x + 1, (int)currentposition.y + 1).a;
        //float IO = texture.GetPixel((int)currentposition.x + 1, (int)currentposition.y ).a;

        //float t_x = currentposition.x - Mathf.Floor(currentposition.x);
        //float t_y = currentposition.y - Mathf.Floor(currentposition.y);

        //float2 AA = Trig(OO,0,0);
        //float2 AB = Trig(OI, 0, 0);
        //float2 BA = Trig(IO, 0, 0);
        //float2 BB = Trig(II, 0, 0);

        //float2 interpolatedValue = (1 - t_x) * (1 - t_y) * AA +
        //                  t_x * (1 - t_y) * BA +
        //                  (1 - t_x) * t_y * AB +
        //                  t_x * t_y * BB;
        //return Mathf.Atan2(interpolatedValue.y,interpolatedValue.x);
        if (majorcolor)
        {
            return findtheta(currentposition);
            //return thetainternal(currentposition, offset);
        }
        else
        {
            return 0;
        }
    }
    private float3[] circles =
        {
            new float3(668, 668, 1),
            new float3(360, 668, 1),
            new float3(500, 360, 1),
        };
    private float4[] lines =
        {
            new float4(1,1,1, 1)
        };
    float findtheta(float2 id)
    {
        const float episilon = 0.0000001f;
        const float scale = 100.0f;

        Vector2 direction = Vector2.zero;

        

        float2 floatingD = new float2((id.x), (id.y));

        for (int c = 0; c < circles.Length; ++c)
        {
            float2 origin = circles[c].xy;
            float decay = circles[c].z;
            Vector2 usingVEC = floatingD - origin;
            Vector2 PreDirection = new Vector2((usingVEC.y * usingVEC.y) - (usingVEC.x * usingVEC.x), -2.0f * usingVEC.x * usingVEC.y).normalized;
            //PreDirection = PreDirection.normalized;

            float distanceaway = usingVEC.magnitude / scale;
            PreDirection = PreDirection * (Mathf.Exp(-(distanceaway * distanceaway) * decay) + episilon);
            direction += PreDirection;
        }
        for (int l = 0; l < lines.Length; ++l)
        {
            float2 origin = lines[l].xy;
            float decay = lines[l].w;
            Vector2 usingVEC = floatingD - origin;
            Vector2 PreDirection = new Vector2(Mathf.Cos(2 * lines[l].z), Mathf.Sin(2 * lines[l].z)).normalized;
            //PreDirection = PreDirection.normalized;

            float distanceaway = usingVEC.magnitude / scale;
            PreDirection = PreDirection * (Mathf.Exp(-(distanceaway * distanceaway) * decay) + episilon);
            direction += PreDirection;
        }

        return Mathf.Atan2(direction.y, direction.x) / 2.0f;
    }
    float thetainternal(float2 currentposition, float2 offset = new float2())
    {
        /*float theta = (Mathf.PerlinNoise(currentposition.x / 66.0f, currentposition.y / 66.0f));
        theta = theta * 0.5f + 0.5f;
        theta = theta * Mathf.PI;// *2;*/

        Vector2 center = new Vector2(256, 256);
        Vector2 direction = ((Vector2)currentposition - (Vector2)center).normalized;


        /*const float stepsize = 0.1f;
        float height = Pnoise(currentposition / spacing);
        Vector3 origin = new Vector3(currentposition.x,currentposition.y,height);

        float right = Pnoise((currentposition + new float2(stepsize,0) ) / spacing);
        Vector3 rightvector = new Vector3(currentposition.x + stepsize, currentposition.y, right) - origin;

        float left = Pnoise((currentposition + new float2(0, stepsize)) / spacing);
        Vector3 leftvector = new Vector3(currentposition.x, currentposition.y + stepsize, left) - origin;

        Vector3 gradient = gradientfromnormal(Vector3.Cross(rightvector, leftvector)).normalized;*/
        //Debug.Log(Mathf.Atan2(gradient.z, gradient.x));
        direction = new float2(noised(currentposition / spacing).yz);
        //direction = new Vector2(gradient.x,gradient.z);

        float theta = Mathf.Atan2(direction.y, direction.x);
        if (direction == Vector2.zero)
        {
            //theta = float.NaN;
        }//theta = Pnoise(currentposition / spacing);
        /*//return theta;
        float Q11 = read1dto2d(currentposition); // current position
        float Q21 = read1dto2d(currentposition + new float2(MathF.Sign(offset.x), 0)); // x corner
        float Q12 = read1dto2d(currentposition + new float2(0, MathF.Sign(offset.y))); // y corner
        float Q22 = read1dto2d(currentposition + new float2(MathF.Sign(offset.x), MathF.Sign(offset.y))); // xy corner

        // Determine the interpolation weights
        float tx = offset.x - (float)Math.Floor(offset.x); // Fractional part of offset.x
        float ty = offset.y - (float)Math.Floor(offset.y); // Fractional part of offset.y

        // Bilinear interpolation
        float R1 = (1 - tx) * Q11 + tx * Q21;
        float R2 = (1 - tx) * Q12 + tx * Q22;
        float theta = (1 - ty) * R1 + ty * R2;//currentposition += new float2(MathF.Ceiling(offset.x),MathF.Ceiling(offset.y));
        //Debug.Log(theta);*/
        return theta;

        //Debug.Log(Derivitive.GetPixel((int)currentposition.x, (int)currentposition.y));
        // Derivitive.GetPixel((int)currentposition.x,(int)currentposition.y).r;
    }
    float read1dto2d(float2 currentposition)
    {
        return data[(int)(currentposition.x * textureSize * scale + currentposition.y)];
    }
    bool Isitlooped(float2 st, List<float2> checker)
    {
        int length = checker.Count;
        for (int i = 0; i < length - minimaldistance * 1; i++)
        {
            float2 position = checker[i];
            if (sqrmagfloat2((st - position), (minimaldistance * minimaldistance) / 1))
            {
                return true;
            }
        }
        return false;
    }
    void tracefromposition(float2 currentposition, Texture2D texture, int major, int sign, out List<float2> tobeadd, float seperationdistance = minimaldistance, bool edgestop = false)
    {
        //float2 origin = currentposition;
        //float theta;
        float2 prevMinorEigenVector = new float2();
        tobeadd = new List<float2>();

        int z;

        for (z = 0; z < 1000000; z++)
        {

            //theta = readtheta(currentposition);

            /*theta = (Mathf.PerlinNoise(currentposition.x / 66.0f, currentposition.y / 66.0f));
            theta = theta * 0.5f + 0.5f;
            theta = theta * Mathf.PI;*/

            //float2 MajorEigenVector = new float2(Mathf.Cos(theta), Mathf.Sin(theta));
            float2 MinorEigenVector = rungekutta(currentposition, major); //new float2(Mathf.Cos(theta + major*Mathf.PI / 2), Mathf.Sin(theta + major * Mathf.PI / 2));
            bool isvalid;
            bool isintercept;
            //bool isint = false;

            if (Vector2.Dot(MinorEigenVector, prevMinorEigenVector) < 0)
            {
                MinorEigenVector = -MinorEigenVector;
            }

            if (major == 0)
            {
                isvalid = ispointvalid(currentposition, major, ref Exisitng, seperationdistance);
                isintercept = ispointvalid(currentposition, major, ref exisitng, 0.81f);
                //isint = ispointvalid(currentposition, major, ref exisitng,1f);
                if (!isintercept)
                {
                    float2 toadd = MinorEigenVector*sign;
                    if (toadd.x < 0)
                    {
                        //toadd = -toadd;
                    }
                    intersections.Add(new float4(currentposition.xy, toadd.xy));
                }
            }
            else
            {
                isvalid = ispointvalid(currentposition, major, ref exisitng, seperationdistance);
                isintercept = ispointvalid(currentposition, major, ref Exisitng, 0.81f);
                if (!isintercept)
                {
                    float2 toadd = MinorEigenVector*sign;
                    if (toadd.x < 0)
                    {
                        //toadd = -toadd;
                    }
                    intersections.Add(new float4(currentposition.xy, new float2(-toadd.y, toadd.x)));
                }
                
                //isint = ispointvalid(currentposition, major, ref Exisitng,1f);
            }

            //if (!isintercept)
            //{
            //    intersections.Add(new float4(currentposition.xy, MinorEigenVector.xy));
            //}



            //if (major == 1)
            //{
            //    MinorEigenVector = new float2(-MinorEigenVector.y, MinorEigenVector.x);
            //}
            
            

            
            //Debug.Log(isvalid);
            /*if (edgestop)
            {
                isvalid = ispointvalid(currentposition, major, ref exisitng, 0.81f) && ispointvalid(currentposition, major, ref Exisitng, 0.81f);
            }*/
            if (z <= 1 && edgestop)
            {
                isvalid = true; isintercept = false;
            }

            
            texture.SetPixel((int)currentposition.x, (int)currentposition.y, roadcol);
            /*if (MathF.Abs(noised(currentposition/spacing).x) < 0.06)
            {
                texture.SetPixel((int)currentposition.x, (int)currentposition.y, new Color());
            }*/



            if (z % minimaldistance / (1) == 0)
            {
                seedclone.Add(currentposition);
            }
            /*if (z% (minimaldistance*3) == 0  && false)
            {
                if (major == 0)
                {
                    modifyspecified(ref Exisitng, tobeadd);
                }
                else
                {
                    modifyspecified(ref exisitng, tobeadd);
                }
                tobeadd.Clear   ();
            }*/


            if (MinorEigenVector.Equals(new float2()) || Isitlooped(currentposition, tobeadd) || !isvalid) // || sqrmagfloat2(currentposition-origin,Mathf.Epsilon)&& texture.GetPixel((int)currentposition.x,(int)currentposition.y) == roadcol)
            {
                //Debug.Log(texture.GetPixel((int)currentposition.x, (int)currentposition.y));
                ends.Add(new float4(currentposition, major, sign));
                break;
                /*if (!isint)
                {
                    Debug.Log("Correct out");
                    
                }*/


            }
            else
            {
                tobeadd.Add(currentposition);
            }
            currentposition -= MinorEigenVector * sign;
            prevMinorEigenVector = MinorEigenVector;
            /*if (!isint)
            {
                roadcol = new Color(0, 1, 1, 1);
            }*/
            //else { Debug.Log(texture.GetPixel((int)currentposition.x, (int)currentposition.y)); }
            //Debug.Log(texture.GetPixel((int)currentposition.x, (int)currentposition.y));


            /*if (z == 100) 
            {
                origin = currentposition;
                //tracefromposition(currentposition, texture, (major+1)%2,sign);
                //tracefromposition(currentposition, texture, (major + 1) % 2, -1);
            }*/


            //exisitng[(int2)(currentposition)] currentposition;
        }

        texture.Apply();
        //Graphics.Blit(texture, heightmapTexture);

        //Dictionary<int2, List<float2>> inuse = exisitng;

        /*if (major == 0)
        {
            modifyspecified(ref Exisitng, tobeadd);
        }
        else
        {
            modifyspecified(ref exisitng, tobeadd);
        }*/



        //currentposition = origin;


        /*theta = readtheta(currentposition);

        major = -(major - 1);
        MinorEigenVector = new float2(Mathf.Cos(theta + major * Mathf.PI / 2), Mathf.Sin(theta + major * Mathf.PI / 2));
        */
        //return origin -= MinorEigenVector * sign*minimaldistance ;

        //yield return new WaitForEndOfFrame();
        //StartCoroutine(tracefromposition(origin, texture, (major + 1) % 2, 1));
        //StartCoroutine(tracefromposition(origin, texture, (major + 1) % 2, -1));
        //tracefromposition(origin, texture, (major + 1) % 2, 1);
        //tracefromposition(origin, texture, (major + 1) % 2, -1);

    }
    bool sqrmagfloat2(float2 vector, float refrence)
    {
        return (vector.x * vector.x + vector.y * vector.y) < refrence;
    }
    void morechecks(ref List<float2> tocheck, int2 direction, int major, int2 pointofintrest)
    {
        float prevcount = tocheck.Count;

        int2 indextocheckforthisone = direction + playerposition;
        pointofintrest = pointofintrest - direction * (textureSize / (int)minimaldistance);
        if (chunkstorage.ContainsKey(indextocheckforthisone))
        {

            if (major == 0)
            {
                if (chunkstorage[indextocheckforthisone].MAJORS.ContainsKey(pointofintrest))
                {
                    //tocheck.AddRange(chunkstorage[indextocheckforthisone].MAJORS[pointofintrest]);
                    foreach (var item in chunkstorage[indextocheckforthisone].MAJORS[pointofintrest])
                    {
                        tocheck.Add(item + (float2)direction * textureSize);
                    }
                }

            }
            else
            {
                if (chunkstorage[indextocheckforthisone].minors.ContainsKey(pointofintrest))
                {
                    //tocheck.AddRange(chunkstorage[indextocheckforthisone].minors[pointofintrest]);
                    foreach (var item in chunkstorage[indextocheckforthisone].minors[pointofintrest])
                    {
                        tocheck.Add(item + (float2)direction * textureSize);
                    }
                }
            }
        }
        /*if ((prevcount - tocheck.Count) != 0)
        {
            Debug.Log(prevcount - tocheck.Count);
        }*/
    }
    bool ispointvalid(float2 point, int major, ref Dictionary<int2, List<float2>> willuse, float seperation = minimaldistance, Vector2 direction = new Vector2())
    {
        //const float seperationdistancetocheck = textureSize / minimaldistance;


        for (int i = -1; i <= 1; i++)
        {
            for (int c = -1; c <= 1; c++)
            {
                Dictionary<int2, List<float2>> inuse = willuse;



                //dictionaryFrom.ToList().ForEach(x => inuse.Add(x.Key, x.Value));

                int2 offset = new int2(i, c);
                /*if (i > seperationdistancetocheck || i <= 0)
                {
                    int j = 0;
                    if (c > seperationdistancetocheck || c <= 0)
                    {
                        j = c;
                    }
                    *//*if (chunkstorage.ContainsKey(playerposition + new int2(i, j)))
                    {
                        Chunk willbsearched = chunkstorage[playerposition + new int2(i,j)];
                        if (major == 0)
                        {
                            inuse = willbsearched.MAJORS;
                            // Existing
                        }
                        else
                        {
                            inuse = willbsearched.minors;
                        }
                    }*//*
                    
                }*/
                int2 pointofintrest = new int2(point / minimaldistance) + offset;
                List<float2> tocheck = new List<float2>();
                if (inuse.ContainsKey(pointofintrest))
                {
                    tocheck = inuse[pointofintrest];


                }

                //float prevcount = tocheck.Count;
                //int2 indextocheckforthisone = new int2(playerposition + new int2(1, 0));
                if (!Vector2.Equals(direction, new Vector2()))
                {
                    List<float2> extra = new List<float2>();
                    morechecks(ref extra, new int2(-1, 0), major, pointofintrest);
                    morechecks(ref extra, new int2(1, 0), major, pointofintrest);
                    morechecks(ref extra, new int2(0, 1), major, pointofintrest);
                    morechecks(ref extra, new int2(0, -1), major, pointofintrest);

                    morechecks(ref extra, new int2(1, 1), major, pointofintrest);
                    morechecks(ref extra, new int2(1, -1), major, pointofintrest);
                    morechecks(ref extra, new int2(-1, 1), major, pointofintrest);
                    morechecks(ref extra, new int2(-1, -1), major, pointofintrest);
                    foreach (Vector2 positon in extra)
                    {
                        if (sqrmagfloat2(positon - (Vector2)point, seperation * seperation) && Mathf.Abs(Vector2.Dot(positon, point)) > 0.5f)
                        {
                            return false;
                        }
                    }
                }
                /*if ((prevcount - tocheck.Count) != 0)
                {
                    Debug.Log(prevcount - tocheck.Count);
                }*/

                foreach (float2 positon in tocheck)
                {
                    if (sqrmagfloat2(positon - point, seperation * seperation))
                    {
                        return false;
                    }
                }


            }
        }



        bool finalcheckup = isinbounds(point);//||(MathF.Abs(noised((point + playerposition * (int)textureSize) / spacing).x) < 0.06)

        if (finalcheckup)
        {
            const float greatervalue = textureSize * scale - maybenot;
            const float offset = -0.1f;
            if (point.x >= greatervalue)
            {
                tostore.Right.Add(new float3(point - new float2(greatervalue + offset, 0), major));
            }
            if (point.y >= greatervalue)
            {
                tostore.Up.Add(new float3(point - new float2(0, greatervalue + offset), major));
            }
            if (point.x <= 0 + maybenot)
            {
                tostore.Left.Add(new float3(point + new float2(greatervalue + offset, 0), major));
            }
            if (point.y <= 0 + maybenot)
            {
                tostore.Down.Add(new float3(point + new float2(0, greatervalue + offset), major));
            }
        }
        return !finalcheckup;
    }
    bool isinbounds(float2 point, float extrastrict = 0)
    {
        return point.x >= textureSize * scale - (maybenot + extrastrict) ||
            point.y >= textureSize * scale - (maybenot + extrastrict) ||
            point.x <= 0 + (maybenot + extrastrict) ||
            point.y <= 0 + (maybenot + extrastrict);
    }
    const int maybenot = 1;
    RenderTexture readrendertxture(string name, TextureFormat format)
    {
        RenderTexture TextureRender = new RenderTexture(textureSize * scale + scale * 0, textureSize * scale + scale * 0, 0, RenderTextureFormat.ARGBFloat);
        TextureRender.enableRandomWrite = true;
        TextureRender.Create();

        // Set parameters for the compute shader
        heightmapComputeShader.SetTexture(0, name, TextureRender);
        return TextureRender;
        /*
                // Dispatch the compute shader
                int threadGroups = Mathf.CeilToInt(textureSize / shaderdimensions);
                //heightmapComputeShader.Dispatch(0, threadGroups * 1, threadGroups, 1);
                // Create a Texture2D with pixel data
                Texture2D texture = new Texture2D(textureSize * scale, textureSize * scale, format, false);
                RenderTexture.active = heightmapTexture;
                texture.ReadPixels(new Rect(0, 0, heightmapTexture.width, heightmapTexture.height), 0, 0);
                texture.Apply();
                return texture;*/
    }
    void convertfromwren(ref Texture2D texture, RenderTexture src, TextureFormat format)
    {
        texture = new Texture2D(textureSize * scale, textureSize * scale, format, false);
        /*RenderTexture.active = heightmapTexture;
        texture.ReadPixels(new Rect(0, 0, heightmapTexture.width, heightmapTexture.height), 0, 0);*/
        texture.Apply();

    }
    private static AsyncGPUReadbackRequest request;
    void OnCompleteReadBack(AsyncGPUReadbackRequest rrequest)
    {
        if (request.hasError == false)
        {
            var data = request.GetData<float>();
        }
        else
        {
            Debug.Log(request.hasError.GetHashCode());
        }
        //Debug.Log(data.Length);
        //Debug.Log("fulfilled");
        //wontcontinue = true;
    }
    //bool wontcontinue;
    private Dictionary<int2, Chunk> chunkstorage = new Dictionary<int2, Chunk>();
    Chunk tostore;
    private static List<float2> seedclone = new List<float2>();
//    private void tensorendering(AsyncGPUReadbackRequest request)
//    {
//        if (renderQuad == null)
//        {
//            return;
//        }
//        if (request.hasError)
//        {
//            Debug.LogError("Readback failed!");
//            return;
//        }
//        //Debug.Log("cameback");
//        Texture2D texture = new Texture2D(textureSize * scale, textureSize * scale, TextureFormat.RGBAFloat, false);
//        texture.LoadRawTextureData(request.GetData<byte>());
//        texture.Apply();/*


        
        
        
//MeshRenderer renderer = renderQuad.GetComponent<MeshRenderer>();

//        renderer.material.mainTexture = texture;*/

//        /*else
//        {
            
//        }*///heightmapTexture = readrendertxture("Result", TextureFormat.RGBAFloat);
//           //Graphics.Blit(texture, heightmapTexture);
//        //Console.WriteLine(renderQuad.name, heightmapTexture);
        
//        callmaster += 1;
//        if (true)//(callmaster <= 1)
//        {
//            heightmapComputeShader.SetTexture(heightmapComputeShader.FindKernel("Lines"), "Result", heightmapTexture);
//            heightmapComputeShader.SetTexture(heightmapComputeShader.FindKernel("Lines"), "inputTexture", texture);
//            heightmapComputeShader.SetInt("mapsize", textureSize * scale);
//            int threadGroups = Mathf.CeilToInt(textureSize / shaderdimensions);
//            heightmapComputeShader.Dispatch(heightmapComputeShader.FindKernel("Lines"), threadGroups, threadGroups, 1);
//            //UnityEngine.Rendering.AsyncGPUReadbackRequest rec = UnityEngine.Rendering.
//                AsyncGPUReadback.Request(heightmapTexture, 0, calcfield);
//        }
//    }
    private void calcfield(AsyncGPUReadbackRequest request)
    {
        if (renderQuad == null)
        {
            return;
        }
        Texture2D texture = new Texture2D(textureSize * scale, textureSize * scale, TextureFormat.RGBAFloat, false);
        texture.LoadRawTextureData(request.GetData<byte>());
        texture.Apply();





        MeshRenderer renderer = renderQuad.GetComponent<MeshRenderer>();

        renderer.material.mainTexture = texture;
        if (proceed)
        {
            StartCoroutine(chunkgenloop());
        }
        
        // Set parameters for the compute shader
        /*heightmapComputeShader.SetTexture(heightmapComputeShader.FindKernel("CSMain"), "Result", heightmapTexture);
        //heightmapComputeShader.SetTexture(0, "Theta", heightmapTexture);

        // Dispatch the compute shader
        int threadGroups = Mathf.CeilToInt(textureSize / shaderdimensions);
        heightmapComputeShader.Dispatch(heightmapComputeShader.FindKernel("CSMain"), threadGroups, threadGroups, 1);
        // Create a Texture2D with pixel data
        if (true)
        {
            AsyncGPUReadback.Request(heightmapTexture, 0, tensorendering);
        }*/
        

    }
    private RenderTexture heightmapTexture;
    private byte callmaster = 0;
    private GameObject renderQuad;

    private void floodAsquare(Texture2D texture, Vector4 item, float iterations, float2 directions)
    {

        const int offset = 1 * (int)minimaldistance / 2;
        //float2 tracepos = new float2(item.x+offset, item.y+offset);
        // texture.SetPixel((int)currentposition.x, (int)currentposition.y, roadcol);
        HashSet<int2> paintbucket = new HashSet<int2>();
        HashSet<int2> drawn = new HashSet<int2>();

        //intersections.Add(new float4(currentposition.xy, MinorEigenVector.xy));
        int2 addition = new int2((int)(item.z * offset *directions.x), (int)(item.w * offset * directions.y));

        paintbucket.Add(new int2((int)(item.x + addition.x - addition.y),(int)(item.y + addition.y + addition.x)));
        //int2 pus = new int2((int)(item.x + addition.x - addition.y), (int)(item.y + addition.y + addition.x));
        
        //new int2((int)(item.x + item.z * offset), (int)(item.y + item.w * offset)) + new int2((int)(-item.w * offset), (int)(item.z * offset));
        //texture.SetPixel((int)pus.x, (int)pus.y, new Color(0, 1, 0, 0));

        int wait = 0;
        while (paintbucket.Count > 0)
        {
            wait += 1;
            //texture.SetPixel((int)pus.x, (int)pus.y, new Color(0, 1, 0, 1));
            HashSet<int2> newpaintbucket = new HashSet<int2>();
            foreach (var position in paintbucket)
            {
                //Debug.Log(texture.GetPixel(position.x, position.y).a);
                drawn.Add(position);
                if (texture.GetPixel(position.x, position.y).r != roadcol.r
                && texture.GetPixel(position.x, position.y).b == 0
                && !isinbounds(position, -1)
                    )
                {
                    //Debug.Log(position);
                    float colorid = 1 - (iterations + 1) / Mathf.Pow((textureSize + 3) / minimaldistance, 2);
                    texture.SetPixel((int)position.x, (int)position.y, new Color(0, colorid, 1, colorid));
                    //return;
                    newpaintbucket.Add(position + new int2(1, 0));
                    newpaintbucket.Add(position + new int2(-1, 0));
                    newpaintbucket.Add(position + new int2(0, 1));
                    newpaintbucket.Add(position + new int2(0, -1));
                }
                
            }
            newpaintbucket.ExceptWith(drawn);
            paintbucket = newpaintbucket;
            //if (iterations > wait )
            //{
            //    iterations = 0;
            //    texture.Apply();
            //    yield return null;
            //}

        }
        //if (iterations > wait)
        //{
        //    iterations = 0;
        //    texture.Apply();
        //    yield return null;
        //}
    }
    IEnumerator WriteToRenderTexture(int2 chunkposition, List<float3> Priority)
    {
        majorcolor = true;
        seedclone.Clear();
        roadcol = new Color(1, 0, 0);
        List<float2> origins = new List<float2>();
        seedclone = new List<float2>();

        intersections.Clear();
        callmaster = 0;
        tostore = new Chunk(
            new List<float3>(),
            new List<float3>(),
            new List<float3>(),
            new List<float3>(),
            exisitng,
            Exisitng,
            new GameObject()
            );
        List<float2> seedQ = new List<float2>();



        //wontcontinue = false;
        Chalted = false;
        exisitng = new Dictionary<int2, List<float2>>();
        Exisitng = new Dictionary<int2, List<float2>>();
        ends.Clear();
        supplements = new Dictionary<int2, List<float2>>();


        
        /*while (!request.done)
        {
            yield return new WaitForEndOfFrame();
        }*/
        
        

        //heightmapTexture = readrendertxture("Result", TextureFormat.RGB24);
        //RenderTexture theta = readrendertxture("Theta", TextureFormat.RFloat);

        //int threadGroups = Mathf.CeilToInt(textureSize / shaderdimensions);

        //ComputeBuffer Buff = new ComputeBuffer(textureSize * scale*textureSize*scale*8*8, sizeof(float));
        //request = AsyncGPUReadback.Request(Buff, OnCompleteReadBack);
        //data = new float[textureSize * scale];
        //heightmapComputeShader.SetBuffer(0, "Theta", Buff);
        //Debug.Log("stage 1");
        //heightmapComputeShader.Dispatch(0, threadGroups * 1, threadGroups, 1);
        //yield return new WaitForSeconds(1f);
        //Buff.GetData(data);
        //Debug.Log("stage 2");
        //yield return new WaitForEndOfFrame();
        // theta
        /*RenderTexture renderTexture = new RenderTexture(textureSize * scale, textureSize * scale, 0, RenderTextureFormat.RFloat);
        renderTexture.enableRandomWrite = true;
        renderTexture.Create();*/


        //Debug.Log(request.hasError.ToString());
        // theta
        //RenderTexture.active = heightmapTexture;

        /*Texture2D texture = new Texture2D(textureSize * scale, textureSize * scale, TextureFormat.RGB24, false);
        // Set the quad's position
        GameObject quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        quad.transform.position = new Vector3(chunkposition.x * reallifescale, 0, chunkposition.y * reallifescale);
        quad.transform.localScale = new Vector3(reallifescale, reallifescale, 1);
        quad.transform.rotation = Quaternion.Euler(90, 0, 0);*/

        // Get the MeshRenderer component
        MeshRenderer renderer = renderQuad.GetComponent<MeshRenderer>();
        Texture2D texture = (Texture2D)renderer.material.mainTexture;
    ////renderer.material.mainTexture = texture;
        //convertfromwren(ref texture, heightmapTexture, TextureFormat.RGB24);
        //convertfromwren(ref Derivitive, theta, TextureFormat.RFloat);

        foreach (float3 seed in Priority)
        {


            bool isvalid;
            if ((int)seed.z == 0)
            {
                isvalid = ispointvalid(seed.xy, (int)seed.z, ref Exisitng);
            }
            else
            {
                isvalid = ispointvalid(seed.xy, (int)seed.z, ref exisitng);
            }



            if (isvalid)
            {
                List<float2> forwardadd = new List<float2>();
                List<float2> backwardadd = new List<float2>();

                tracefromposition(seed.xy, texture, (int)seed.z, 1, out forwardadd);
                tracefromposition(seed.xy, texture, (int)seed.z, -1, out backwardadd);

                if ((int)seed.z == 0)
                {
                    modifyspecified(ref Exisitng, forwardadd);
                    modifyspecified(ref Exisitng, backwardadd);
                }
                else
                {
                    modifyspecified(ref exisitng, forwardadd);
                    modifyspecified(ref exisitng, backwardadd);
                }




                seedQ.AddRange(forwardadd);
                seedQ.AddRange(backwardadd);
                //seedclone.AddRange(forwardadd);
                //seedclone.AddRange(backwardadd);
            }
            yield return null;
        }
        Priority.Clear();
        //Debug.Log(seedQ.Count);
        //Dictionary<float2,float2> exisitng = new Dictionary<float2,float2>();

        //int currentmajor = 0;
        float2 currentposition = new float2(1 * scale * textureSize / 2, scale * textureSize / 2);
        float2 primpos = currentposition;
        //yield return new WaitForEndOfFrame();
        /*while (wontcontinue)
        {
            
        }*/
        //Debug.Log("made it to loop");
        while (seedQ.Count > 0)
        {
            currentposition = seedQ[0];
            //yield return new WaitForEndOfFrame();
            const bool isvalid = true;
            if (ispointvalid(currentposition, 0, ref Exisitng,StartWider))
            {
                intersections.Add(new float4(currentposition.xy, rungekutta(currentposition, 0)));
                const int currentmajor = 0;
                //isvalid = ispointvalid(currentposition, currentmajor,ref Exisitng);
                List<float2> forwardadd = new List<float2>();
                List<float2> backwardadd = new List<float2>();
                if (isvalid)
                {
                    tracefromposition(currentposition, texture, currentmajor, 1, out forwardadd);
                    tracefromposition(currentposition, texture, currentmajor, -1, out backwardadd);
                }

                modifyspecified(ref Exisitng, forwardadd);
                modifyspecified(ref Exisitng, backwardadd);


                seedQ.AddRange(forwardadd);
                seedQ.AddRange(backwardadd);
                //seedclone.AddRange(forwardadd);
                //seedclone.AddRange(backwardadd);
                yield return null;
                //yield return new WaitForSeconds(1);
            }
            else if (ispointvalid(currentposition, 1, ref exisitng,StartWider))
            {
                intersections.Add(new float4(currentposition.xy, rungekutta(currentposition, 0)));
                const int currentmajor = 1;
                //isvalid = ispointvalid(currentposition, currentmajor, ref exisitng);
                List<float2> forwardadd = new List<float2>();
                List<float2> backwardadd = new List<float2>();
                if (isvalid)
                {
                    tracefromposition(currentposition, texture, currentmajor, 1, out forwardadd);
                    tracefromposition(currentposition, texture, currentmajor, -1, out backwardadd);
                }

                modifyspecified(ref exisitng, forwardadd);
                modifyspecified(ref exisitng, backwardadd);

                seedQ.AddRange(forwardadd);
                seedQ.AddRange(backwardadd);
                //seedclone.AddRange(forwardadd);
                //seedclone.AddRange(backwardadd);
                yield return null;
                //yield return new WaitForSeconds(1);
            }
            /*else
            {
                isvalid = false;
            }*/



            //Debug.Log(isvalid + " " + seedQ.Count);
            /*List < float2 > forwardadd = new List<float2>();
            List<float2> backwardadd = new List<float2>();
            if (isvalid )
            {
                tracefromposition(currentposition, texture, currentmajor, 1, out forwardadd);
                tracefromposition(currentposition, texture, currentmajor, -1, out backwardadd);
            }
            if (currentmajor == 0)
            {
                modifyspecified(ref Exisitng, forwardadd);
                modifyspecified(ref Exisitng, backwardadd);
            }
            else
            {
                modifyspecified(ref exisitng, forwardadd);
                modifyspecified(ref exisitng, backwardadd);
            }
            seedQ.AddRange(forwardadd);
            seedQ.AddRange(backwardadd);*/

            seedQ.Remove(currentposition);
            //Debug.Log(seedQ.Count);

            //yield return new WaitForEndOfFrame();
            //seedQ = new List<float2>();
            //currentmajor = -(currentmajor - 1);
        }

        // identify shapes

        // below is the atrociously slow version
        float iterations = 0;
        foreach (var item in intersections)
        {
            //Debug.Log(item);
            
            iterations += 1;
            floodAsquare(texture, item,iterations,new float2(1,1));
            floodAsquare(texture, item, iterations, new float2(1, -1));
            floodAsquare(texture, item, iterations, new float2(-1, 1));
            floodAsquare(texture, item, iterations, new float2(-1, -1));

            if (iterations % 100 == 0)
            {
                yield return null; // take a break
                texture.Apply();
            }
        }
        texture.Apply();
        //Texture2D empty = new Texture2D(textureSize * scale, textureSize * scale, TextureFormat.RGBAFloat, false);
        float3[] newcircle =
        {
            new float3(100,100,1)
        };
        circles = newcircle;
        LICsetup(texture);
        Debug.Log("done");



        //List<float4> actual = new List<float4>();

        //Debug.Log(exisitng.Count);
        tostore.minors = new Dictionary<int2, List<float2>>(exisitng);
        tostore.MAJORS = new Dictionary<int2, List<float2>>(Exisitng);
        //exisitng.Clear();
        //Exisitng.Clear();
        //Debug.Log(tostore.minors.Count);
        roadcol = new Color(0, 0, 0);

        seedQ = seedclone;
        majorcolor = false;
        //Debug.Log(seedQ.Count + " " + seedclone.Count);
        //seedclone.Clear();
        while (seedQ.Count > 0 && false)
        {
            currentposition = seedQ[0];
            //yield return new WaitForEndOfFrame();
            const bool isvalid = true;
            if (ispointvalid(currentposition, 0, ref Exisitng, majormindis+StartWider))
            {
                const int currentmajor = 0;
                //isvalid = ispointvalid(currentposition, currentmajor,ref Exisitng);
                List<float2> forwardadd = new List<float2>();
                List<float2> backwardadd = new List<float2>();
                if (isvalid)
                {
                    tracefromposition(currentposition, texture, currentmajor, 1, out forwardadd, majormindis);
                    tracefromposition(currentposition, texture, currentmajor, -1, out backwardadd, majormindis);
                }

                modifyspecified(ref Exisitng, forwardadd);
                modifyspecified(ref Exisitng, backwardadd);


                seedQ.AddRange(forwardadd);
                seedQ.AddRange(backwardadd);
                //seedclone.AddRange(forwardadd);
                //seedclone.AddRange(backwardadd);
                yield return null;
                //yield return new WaitForSeconds(1);
            }
            else if (ispointvalid(currentposition, 1, ref exisitng, minormindis+StartWider))
            {
                const int currentmajor = 1;
                //isvalid = ispointvalid(currentposition, currentmajor, ref exisitng);
                List<float2> forwardadd = new List<float2>();
                List<float2> backwardadd = new List<float2>();
                if (isvalid)
                {
                    tracefromposition(currentposition, texture, currentmajor, 1, out forwardadd, minormindis);
                    tracefromposition(currentposition, texture, currentmajor, -1, out backwardadd, minormindis);
                }

                modifyspecified(ref exisitng, forwardadd);
                modifyspecified(ref exisitng, backwardadd);

                seedQ.AddRange(forwardadd);
                seedQ.AddRange(backwardadd);
                //seedclone.AddRange(forwardadd);
                //seedclone.AddRange(backwardadd);
                yield return null;
                //yield return new WaitForSeconds(1);
            }
            /*else
            {
                isvalid = false;
            }*/



            //Debug.Log(isvalid + " " + seedQ.Count);
            /*List < float2 > forwardadd = new List<float2>();
            List<float2> backwardadd = new List<float2>();
            if (isvalid )
            {
                tracefromposition(currentposition, texture, currentmajor, 1, out forwardadd);
                tracefromposition(currentposition, texture, currentmajor, -1, out backwardadd);
            }
            if (currentmajor == 0)
            {
                modifyspecified(ref Exisitng, forwardadd);
                modifyspecified(ref Exisitng, backwardadd);
            }
            else
            {
                modifyspecified(ref exisitng, forwardadd);
                modifyspecified(ref exisitng, backwardadd);
            }
            seedQ.AddRange(forwardadd);
            seedQ.AddRange(backwardadd);*/

            seedQ.Remove(currentposition);
            //Debug.Log(seedQ.Count);

            //yield return new WaitForEndOfFrame();
            //seedQ = new List<float2>();
            //currentmajor = -(currentmajor - 1);
        }
        while (ends.Count > 0 && false)
        {
            yield return null;
            currentposition = new float2(ends[0].x, ends[0].y);
            //Debug.Log(ends.Count);
            List<float2> result = new List<float2>();

            bool isvalid = ispointvalid(currentposition, (int)ends[0].z, ref supplements, minimaldistance / 3f);
            if (isvalid)
            {

                finishoff(currentposition, texture, (int)ends[0].z, (int)ends[0].w, out result, roadcol);

                if (ends[0].z == 1)
                {
                    modifyspecified(ref exisitng, result);
                }
                else
                {
                    modifyspecified(ref Exisitng, result);
                }
                modifyspecified(ref supplements, result);
                //actual.Add(ends[0]);
            }
            if ((!isvalid || result.Count == 0) && !(currentposition.x >= textureSize * scale - maybenot || currentposition.y >= textureSize * scale - maybenot || currentposition.x <= 0 + maybenot || currentposition.y <= 0 + maybenot))
            {
                const int i = 0;
                const int c = 0;
                //for (int i = -1; i <= 1; i++)
                //{
                //for (int c = -1; c <= 1; c++)
                //{
                finishoff(currentposition + new float2(i, c), texture, (int)ends[0].z, -(int)ends[0].w, out result, new Color(0, 1, 0)); //new Color(0.804f, 0.804f, 0.804f, 0.804f)
                                                                                                                                         //exisitng = supplements.Except(result).ToList();
                removespecified(ref supplements, result);
                removespecified(ref exisitng, result);
                removespecified(ref Exisitng, result);
                //}
                //}


            }

            ends.Remove(ends[0]);
        }

        /*exisitng = new Dictionary<int2, List<float2>>();
        Exisitng = new Dictionary<int2, List<float2>>();*/
        ends.Clear();
        supplements = new Dictionary<int2, List<float2>>();
        //data = new float[0];
        //heightmapTexture.Release();
        //Buff.Release();
        Chalted = true;




        seedQ.Clear();

        /*foreach (var item in intersections)
        {
            GameObject intersection = GameObject.CreatePrimitive(PrimitiveType.Cube);
            intersection.transform.position = new Vector3(-reallifescale / 2,0, -reallifescale / 2) + new Vector3(item.x, 0, item.y)/ realifescalefactor;
            intersection.transform.rotation = Quaternion.LookRotation(new Vector3(item.z,0,item.w));
            intersection.transform.localScale = Vector3.one/6;
            //yield return null;
        }*/
        seedclone.Clear();
        chunkstorage[chunkposition] = tostore;

        //Debug.Log("complete " + seedQ.Count);
    }
    //private MeshCollider placeholder;
    const float realifescalefactor = 16;
    const float reallifescale = textureSize / realifescalefactor;
    private static int2 playerposition;
    private static bool proceed = true;
    IEnumerator chunkgenloop()
    {
        
        while (proceed)
        {
            proceed = false;
            if (Chalted)
            {
                playerposition = new int2(Mathf.RoundToInt(transform.position.x / reallifescale), Mathf.RoundToInt(transform.position.z / reallifescale));
                //Debug.Log(chunkstorage.ContainsKey(playerposition) + " " + playerposition);
                if (!chunkstorage.ContainsKey(playerposition))
                {
                    List<float3> Priority = new List<float3>();
                    const float sparseness = 6;
                    const float distancefromedge = 3;
                    int2 offset = playerposition + new int2(1, 0);
                    if (chunkstorage.ContainsKey(offset))
                    {
                        Priority.AddRange(chunkstorage[offset].Left);
                    }
                    else
                    {
                        for (int c = 0; c < textureSize / (minimaldistance * sparseness); c++)
                        {
                            Priority.Add(new float3(distancefromedge, c * (minimaldistance * sparseness), 1));
                        }
                    }
                    offset = playerposition + new int2(-1, 0);
                    if (chunkstorage.ContainsKey(offset))
                    {
                        Priority.AddRange(chunkstorage[offset].Right);
                    }
                    else
                    {
                        for (int c = 0; c < textureSize / (minimaldistance * sparseness); c++)
                        {
                            Priority.Add(new float3(textureSize - distancefromedge, c * (minimaldistance * sparseness), 1));
                        }
                    }
                    offset = playerposition + new int2(0, 1);
                    if (chunkstorage.ContainsKey(offset))
                    {
                        Priority.AddRange(chunkstorage[offset].Down);
                    }
                    else
                    {
                        for (int c = 0; c < textureSize / (minimaldistance * sparseness); c++)
                        {
                            Priority.Add(new float3(c * (minimaldistance * sparseness), textureSize - distancefromedge, 1));
                        }
                    }
                    offset = playerposition + new int2(0, -1);
                    if (chunkstorage.ContainsKey(offset))
                    {
                        Priority.AddRange(chunkstorage[offset].Up);
                    }
                    else
                    {
                        for (int c = 0; c < textureSize / (minimaldistance * sparseness); c++)
                        {
                            Priority.Add(new float3(c * (minimaldistance * sparseness), distancefromedge, 1));
                        }
                    }
                    //Priority.Clear();
                    /*if (Priority.Count <= 0 || true)
                    {
                        const float sparseness = 6;
                        for (int c = 0; c < textureSize / (minimaldistance * sparseness); c++)
                        {
                            Priority.Add(new float3(c * (minimaldistance * sparseness),3, 1));
                            Priority.Add(new float3( c * (minimaldistance * sparseness), textureSize - 3, 1));
                            Priority.Add(new float3(c * (minimaldistance * sparseness),3, 0));
                            Priority.Add(new float3(c * (minimaldistance * sparseness), textureSize - 3, 0));

                        }
                        for (int c = 0; c < textureSize / (minimaldistance * sparseness); c++)
                        {
                            Priority.Add(new float3(3, c * (minimaldistance * sparseness), 1));
                            Priority.Add(new float3(textureSize-3, c * (minimaldistance * sparseness), 1));
                            Priority.Add(new float3(3, c * (minimaldistance * sparseness), 0));
                            Priority.Add(new float3(textureSize - 3, c * (minimaldistance * sparseness), 0));
                        }
                    }*/

                    yield return StartCoroutine(WriteToRenderTexture(playerposition, Priority));
                }
            }
            yield return new WaitForSeconds(1);
        }
    }
    private void LICsetup(Texture2D input)
    {
        if (renderQuad == null)
        {
            renderQuad = GameObject.CreatePrimitive(PrimitiveType.Quad);
            renderQuad.transform.position = new Vector3(0, 10, 0);
            renderQuad.transform.rotation = Quaternion.Euler(90, 0, 0);
            renderQuad.transform.localScale = new Vector3(textureSize / 8, textureSize / 8, 1);
        }
        



        heightmapTexture = new RenderTexture(textureSize * scale + scale * 0, textureSize * scale + scale * 0, 0, RenderTextureFormat.ARGBFloat);
        heightmapTexture.enableRandomWrite = true;
        heightmapTexture.Create();

        // Set parameters for the compute shader


        ComputeBuffer CirclesBuffer = new ComputeBuffer(circles.Length, sizeof(float)*3);
        CirclesBuffer.SetData(circles);
        //heightmapComputeShader.SetConstantBuffer("circles", CirclesBuffer,0,circles.Length*sizeof(float)*3);
        heightmapComputeShader.SetBuffer(heightmapComputeShader.FindKernel("Lines"), "circles", CirclesBuffer);
        heightmapComputeShader.SetInt("circleLen", circles.Length);

        ComputeBuffer LinesBuffer = new ComputeBuffer(lines.Length, sizeof(float) * 4);
        LinesBuffer.SetData(lines);
        //heightmapComputeShader.SetConstantBuffer("circles", CirclesBuffer,0,circles.Length*sizeof(float)*3);
        heightmapComputeShader.SetBuffer(heightmapComputeShader.FindKernel("Lines"), "lines", LinesBuffer);
        heightmapComputeShader.SetInt("linesLen", lines.Length);
        heightmapComputeShader.SetTexture(heightmapComputeShader.FindKernel("Lines"), "inputTexture", input);

        heightmapComputeShader.SetTexture(heightmapComputeShader.FindKernel("Lines"), "Result", heightmapTexture);
        //heightmapComputeShader.SetTexture(0, "Theta", heightmapTexture);

        // Dispatch the compute shader
        int threadGroups = Mathf.CeilToInt(textureSize / shaderdimensions);
        heightmapComputeShader.Dispatch(heightmapComputeShader.FindKernel("Lines"), threadGroups, threadGroups, 1);
        // Create a Texture2D with pixel data

        UnityEngine.Rendering.AsyncGPUReadbackRequest request = UnityEngine.Rendering.AsyncGPUReadback.Request(heightmapTexture, 0, calcfield);

    }

    //public float[] data;
    void Start()
    {
        /*seedQ.Add(new float2(6,6));
        float2 coolpositions = new float2(1 * scale * textureSize/2 , scale * textureSize /2);
        seedQ.Add(coolpositions);
        coolpositions = new float2(6, scale * textureSize /2);
        seedQ.Add(coolpositions);
        coolpositions = new float2(1 * scale * textureSize /2 , 6);
        seedQ.Add(coolpositions);
        seedQ.Add(new float2(668, 360));*/

        //float2 chunkposition = new float2();


        /*// Initialize the RenderTexture
        heightmapTexture = new RenderTexture(textureSize* scale + scale*0, textureSize* scale + scale*0, 0, RenderTextureFormat.ARGBFloat);
        heightmapTexture.enableRandomWrite = true;
        heightmapTexture.Create();

        // Set parameters for the compute shader
        heightmapComputeShader.SetTexture(0, "Result", heightmapTexture);

        // Dispatch the compute shader
        int threadGroups = Mathf.CeilToInt(textureSize / shaderdimensions);
        heightmapComputeShader.Dispatch(0, threadGroups* 1, threadGroups, 1);*/
        //StartCoroutine(chunkgenloop());
        Texture2D empty = new Texture2D(textureSize * scale, textureSize * scale, TextureFormat.RGBAFloat, false);
        LICsetup(empty);
        /*if (Chalted)
        {
            StartCoroutine(WriteToRenderTexture(new int2()));
        }*/


        // Apply the generated heightmap (Optional)
        //float[] heightmapData = new float[textureSize * textureSize];
        //RenderTexture.active = heightmapTexture;
    }

    // Update is called once per frame
    /*void Update()
    {
        int kernelHandle = 0;// generation.FindKernel("RoadMap");
        
        if (renderTexture == null)
        {
            renderTexture = new RenderTexture(imageresolution, imageresolution, 0, RenderTextureFormat.RFloat);
            renderTexture.enableRandomWrite = true;
            renderTexture.Create();
        }
        generation.SetTexture(kernelHandle, "Tensors", renderTexture);

        int threadGroups = Mathf.CeilToInt(imageresolution / shaderdimensions);

        //Array data = data(0);
        //data = new float[imageresolution* imageresolution];

        /*float[] heightmapData = new float[imageresolution * imageresolution];
        //RenderTexture.active = heightmapTexture;
        Texture2D tempTexture = new Texture2D(imageresolution, imageresolution, TextureFormat.RFloat, false);
        tempTexture.ReadPixels(new Rect(0, 0, imageresolution, imageresolution), 0, 0);
        RenderTexture.active = null;

        
        
        ComputeBuffer tensors = new ComputeBuffer(imageresolution* imageresolution, sizeof(float));
        tensors.SetData(data);

generation.SetBuffer(kernelHandle,"Tensors", tensors);/
        generation.Dispatch(kernelHandle, threadGroups, threadGroups, 1);

        

        //texture.texture = tempTexture;
    }*/
}
public struct Chunk
{
    public Chunk(List<float3> up, List<float3> down, List<float3> left, List<float3> right, Dictionary<int2, List<float2>> MINORS, Dictionary<int2, List<float2>> majors, GameObject models)
    {
        Up = up;
        Down = down;
        Left = left;
        Right = right;
        Models = models;
        minors = MINORS;
        MAJORS = majors;
    }

    public GameObject Models { get; }

    public List<float3> Up { get; }
    public List<float3> Down { get; }
    public List<float3> Left { get; }
    public List<float3> Right { get; }

    public Dictionary<int2, List<float2>> minors { get; set; }
    public Dictionary<int2, List<float2>> MAJORS { get; set; }
}

