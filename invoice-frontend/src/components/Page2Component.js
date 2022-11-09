import React, { useEffect, useState } from "react";
import { TestData } from "../data";
import { TestData2 } from "../data";
import { TestData3 } from "../data";
import SpinnerComponent from "./SpinnerComponent";
import original from "../img/original.png";
import img1 from "../img/img1.png";
import img2 from "../img/img2.png"; 
import ZoomComponent from "./ZoomComponent";
import TableComponent from "./TableComponent";

import axios from "axios";

import Button from "react-bootstrap/Button";
import Modal from "react-bootstrap/Modal";
import ImageCropper from "./ImageCropperComponent";
import RectangleComponent from "./RectangleComponent";
import { Link } from "react-router-dom";

const Page2Component = (props) => {
  let label = ["Invoice number", "Address", "Date", "Total","Category","Bill of Materials"];
  let label1 = ["invono", "address", "date", "total","Category","table"];
  const [crop, setCrop] = useState(null);
  const [prevTxt, setPrevTxt] = useState('');
  const [prevScore, setPrevScore] = useState('');
  const [rect, setRect] = useState(false);
  const [cancel, setCancel] = useState(false);
  const [bill , setBill] = useState(false);
  const [prev, setPrev] = useState(null);
  const [predicted, setPredicted] = useState(true);
  const [imageFile,setImageFile] = useState(null)
  const [data, setData] = useState({
    address: ["", ""],

    date: [{text:""}, ""],
    invono: [{text:""}, ""],

    total: [{text:""}, ""],
    image: "",
    category:[
      "",
      ""
    ],
    headers:[],
    values:[]
  });
  const [edit, setEdit] = useState(-1);
  const [extract, setExtract] = useState('');
  const [collapse, setCollapse] = useState(false);
  const [disable, setDisable] = useState(false);
  const [spin, setSpin] = useState(false);
  const [dat, setDat] = useState(1);
  const [txt, setTxt] = useState("");
  let img = props.img;
  let setImg = props.setImg;
  const [zoom, setZoom] = useState(false);
  let setCoordinates = props.setCoordinates;
  let coordinates = props.coordinates;

  const [fullscreen, setFullscreen] = useState(true);
  const [show, setShow] = useState(false);

  const [currh, setCurrh] = useState(0);
  const [currw, setCurrw] = useState(0);

  function handleShow(breakpoint) {
    setFullscreen(breakpoint);
    setShow(true);
  }

  const setBg = (index) => {
    if (index == 0) {
      return data.invono[0].text != "" && data.invono[1] > 0.6
        ? { background: "green" }
        : { background: "red" };
    }
    if (index == 1) {
      return data.address[0] != "" && data.address[1] > 0.6
        ? { background: "green" }
        : { background: "red" };
    }
    if (index == 2) {
      return data.date[0].text != "" && data.date[1] > 0.6
        ? { background: "green" }
        : { background: "red" };
    }
    if (index == 3) {
      return data.total[0].text != "" && data.total[1] > 0.6
        ? { background: "green" }
        : { background: "red" };
    }
    return { background: "green" };
  };

  const predictlabel = async (event) => {
    setSpin(true);
    console.log(props.file);
    event.preventDefault();
    console.log(img);
    const formData1 = new FormData();
    formData1.append("files[]", props.file);
    let res = { data: {  address: ["", ""],

    date:  [{text:""}, ""],
    invono: [{text:""}, ""],

    total: [{text:""}, ""],
    image: "",
    
    category:[
      "",
      ""
    ],
    headers:[],
    values:[] } };
    //res.data=TestData3
    try {
      res = await axios.post('http://127.0.0.1:5000/predict', formData1);
    } catch (error) {
      window.alert("Some thing went wrong please try again")
      //window.location.reload();
    }
    // const byteCharacters = atob(res.data.image);
    // const byteNumbers = new Array(byteCharacters.length);
    // for (let i = 0; i < byteCharacters.length; i++) {
    //   byteNumbers[i] = byteCharacters.charCodeAt(i);
    // }
    // const byteArray = new Uint8Array(byteNumbers);

    // let image = new Blob([byteArray], { type: "image/jpeg" });
    // let imageUrl = URL.createObjectURL(image);
    // res.data.image = byteCharacters;
    // var image = new Image();
    // image.src = `data:image/png;base64,${res.data.image}`;
    // res.data.image = image;------

    console.log("RES",res);
   
    setDat(dat + 1);
    console.log(dat);
    event.preventDefault();

    setImg(`data:image/png;base64,${res.data.image}`);
    //setImg(img1)

    setData(res.data);

    //setData(TestData3)
    setSpin(false);
    setPredicted(false);
    
  };

  useEffect(() => {
    console.log(data);
  }, [data]);
  const handleSubmit = async () => {
    setZoom(true);

    handleClose();
    setSpin(true);
    console.log("Helll", crop);
    var arr = crop.split(','), mime = arr[0].match(/:(.*?);/)[1],
    bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
    while(n--){
    u8arr[n] = bstr.charCodeAt(n);
    }
  let imageFile1= new File([u8arr], "cropped.png", {type:"image/png"});
  setImageFile(crop);

    var image = new Image();
    image.src = crop;
    //let im = URL.createObjectURL(crop);
    console.log("Imageee",imageFile1);
    const formData2 = new FormData();
    formData2.append("crop[]", imageFile1);
    formData2.append("label", label1[edit]);
    // var reader = new FileReader();
    // reader.readAsDataURL(image); 
    // reader.onloadend = function() {
    //   var base64data = reader.result;                
    //   console.log("Base64",crop);
    // }
    console.log("BBBB")
    console.log( crop)
    let res={data:{text:""}}
    if(edit===5){
      res = { data: {headers:[],values:[]} };
    }
   
     try {
      res = await axios.post("http://127.0.0.1:5000/crop", formData2); 
     } catch (error) {
      window.alert("Some thing went wrong please try again")
     }
     console.log("Cropped",res)
   
    const txt = res.data.text;
    //const txt = "26/09/2022"
    setExtract(txt)
    console.log(res.data);
    if (edit == 0) {
      let temp = data;
      temp.invono= [{text:""}, ""]
      temp.invono[0].text = txt;
      temp.invono[1] = "0.99";
      setTxt(txt);
      setData(temp);
    }
    if (edit == 1) {
      let temp = data;
      console.log(temp)
      setExtract(res.data)
      temp.address[0] = res.data;
      temp.address[1] = "0.99";
      setTxt(txt);
      setData(temp);
    }
    if (edit == 2) {
      let temp = data;
      console.log(temp)
      temp.date= [{text:""}, ""]
      temp.date[0].text = txt;
      temp.date[1] = "0.99";
      setTxt(txt);
      setData(temp);
      console.log("DDDDATA",data)
    }
    if (edit == 3) {
      let temp = data;
      temp.total= [{text:""}, ""]
      setTxt(txt);
      temp.total[0].text = txt;
      temp.total[1] = "0.99";
      setData(temp);
    }
    if (edit == 5) {
      let temp = data;
      setTxt(txt);
      temp.headers = res.data;
      temp.values = res.values;
      setData(temp);
    }
   
    setRect(true)
    setPrev(img);
    setSpin(false);
  };
  const revert = () => {setImg(prev);
    if (edit == 0) {
      let temp = data;
      temp.invono[0].text = prevTxt;
      temp.invono[1] = prevScore;
      setData(temp);
    }
    if (edit == 1) {
      let temp = data;
      temp.address[0] = prevTxt;
      temp.address[1] = prevScore;
      setData(temp);
    }
    if (edit == 2) {
      let temp = data;
      temp.date[0].text = prevTxt;
      temp.date[1] = prevScore;
      setData(temp);
    }
    if (edit == 3) {
      let temp = data;
      temp.total[0].text = prevTxt;
      temp.total[1] = prevScore;
      setData(temp);
    }};
  const handleClose = () => setShow(false);
  const add = (text) => {
    const myArray = text.split("\n");
    console.log(myArray);
    return (
      <>
        {myArray.map((str, index) => {
          if (index === 0) {
            return (
              <p style={{ marginBottom: "0" }}>
                {str}
                {","}
                <span
                  class="ColBtn"
                  type="button"
                  data-toggle="collapse"
                  data-target="#collapseExample"
                  aria-expanded="false"
                  aria-controls="collapseExample"
                  onClick={() => setCollapse(!collapse)}
                >
                  <iconify-icon icon="akar-icons:arrow-up"></iconify-icon>
                </span>
              </p>
            );
          } else {
            if (str != "" && str!="\f") {
              return (
                <p style={{ marginBottom: "0" }}>
                  {str} {myArray[index+1] === '\f' ? "." : ","}
                </p>
              );
            }
            return <></>;
          }
        })}
      </>
    );
  };
  return (
    <>
      <Modal show={show} fullscreen={fullscreen} onHide={() => setShow(false)}>
        <Modal.Header closeButton>
          <Modal.Title>
            <button className="Predictbtn" onClick={handleSubmit}>
              Extract Text
            </button>
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <ImageCropper
            img={img}
            setCoordinates={props.setCoordinates}
            setCrop={setCrop}
            currh={currh}
            currw={currw}
            setCurrh={setCurrh}
            setCurrw={setCurrw}
          /> 
          
        </Modal.Body>
      </Modal>
      {spin ? (
        <>
          <SpinnerComponent />
        </>
      ) : (
        <>
          <fieldset style={{ padding: "auto" }}>
            <div class="container" style={{ minHeight: "731px" }}>
              <div class="row">
                <div class="col">
                  <div class="">
                  
                    <div class="">
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                        }}
                      >
                        {disable && edit!==4 ? (
                          <>
                            {" "}
                            <h5 class="card-title" style={{fontFamily: `Arial, "Helvetica Neue", Helvetica, sans-serif`,fontStyle:"italic"}}>
                              Select value for {label[edit]}
                            </h5>
                            <Button
                             style={{ color: "#ffffff",
                             border: "1px solid #f87115",
                             borderRadius: "15px",
                             padding: "8px 25px",
                             textTransform: "uppercase",
                             fontSize: "13px",
                             fontWeight: "500",
                             letterSpacing: "1px",
                             background: "#f87115"
                             }}
                              onClick={() => handleShow("xxl-down")}
                            >
                              Select Text
                            </Button>
                           
                          </>
                        ) : (
                          <></>
                        )}
                      </div>
                      <br/>
                      {rect && (
                        <RectangleComponent
                          img={img}
                          coordinates={coordinates}
                          setImg={setImg}
                          setMyFinal={props.setMyFinal}
                          currh={currh}
                          currw={currw}
                          setCurrh={setCurrh}
                          setCurrw={setCurrw}
                          txt={label[edit]}
                          setRect={setRect}
                          revert={revert}
                          extract={extract}
                          label={label[edit]}
                          cancel={cancel}
                          setCancel={setCancel}
                        />
                      ) 
}
                      {
                        img && ( 
                          <div class="member" data-aos="zoom-in" data-aos-delay="100">
                          <div class="about-img">
                          <img
                          class="card-img-top"
                          src={img}
                          alt="Card image cap"
                        />
                         <div class="member-info">
                <div class="member-info-content">
                  <h4> <button
                        type="button"
                        class="Predictbtn"
                        onClick={() => {
                          props.setIndex(0);
                          window.location.reload()
                        }}
                      >
                        Change File
                      </button></h4>
                  <span>{props.name}</span>
                </div>
              
              </div>
                          </div>
                        </div>
                        )
                      }
                    </div>
                   
                  </div>
                </div>
                <div class="Tabcol col card" style={{ border: "none" }}>
                  {predicted ? (
                    <div className="Empt">
                      <p>
                        Tap the button to extract the label and text ...!
                      </p>
                      <div class="col-lg-4 d-flex align-items-center justify-content-center position-relative" style={{width: "100%"}} data-aos="zoom-in" data-aos-delay="200">
                      <button className="glightbox play-btn" onClick={predictlabel}>
                        {" "}
                       
                      </button>
                      </div>
                    </div>
                  ) : (
                    <>
                    <div style={bill ? {display:"none"} : {} }  >
                      <p className="TabHead">
                        Predicted L<span>ables</span>
                      </p>
                      <div className="card-body Table">
                        <table class="Tab ">
                          <thead>
                            <th class="headcol"> <span style={{backgroundolor: "white"}}>Label</span> </th>
                            <th>Text</th>
                            <th>Score</th>
                            <th >
                              {" "}
                              <span className="Pen"><iconify-icon  icon="uil:pen"></iconify-icon>{" "}</span> 
                            </th>
                          </thead>
                          <tbody>
                            <tr>
                              <td class="headcol">
                                <span class="dot" style={setBg(0)}></span>
                                Invoice No
                              </td>
                              <td>
                                <input
                                  style={{ border: "none" }}
                                  className="typing-container"
                                  value={data.invono[0].text}
                                  onChange={(event) => {
                                    let temp = data;
                                    temp.invono[0].text = event.target.value;
                                    setTxt(event.target.value);
                                    setData(temp);
                                    console.log(temp.invono[0].text);
                                  }}
                                  readOnly={
                                    edit === 0 && disable ? false : true
                                  }
                                />
                              </td>
                              <td>
                                {data.invono[1] == "" ? "---" : data.invono[1]}
                              </td>
                              <td >
                                <button
                                className="Pen"
                                  onClick={(e) => {
                                    e.preventDefault();
                                    setPrevTxt(data.invono[0].text)
                                    setPrevScore(data.invono[1])
                                    setEdit(0);
                                    setDisable(!disable);
                                    console.log(disable);
                                  }}
                                >
                                  {edit === 0 && disable ? (
                                    <iconify-icon  icon="charm:tick"></iconify-icon>
                                  ) : (
                                    <iconify-icon  icon="uil:pen"></iconify-icon>
                                  )}
                                </button>
                              </td>
                            </tr>
                            <tr>
                              <td className="Add headcol" >
                                <span class="dot" style={setBg(1)}></span>
                                Address
                              </td>
                              <td>
                                {" "}
                                <p
                                  style={
                                    collapse
                                      ? {
                                          display: "none",
                                        }
                                      : { marginBottom: "0" }
                                  }
                                >
                                  {data.address[0] == ""
                                    ? "---"
                                    : data.address[0] && data.address[0].substring(0, 6) +
                                      "  ...  "}

                                  <span
                                    class="ColBtn"
                                    type="button"
                                    data-toggle="collapse"
                                    data-target="#collapseExample"
                                    aria-expanded="false"
                                    aria-controls="collapseExample"
                                    onClick={() => setCollapse(!collapse)}
                                  >
                                    <iconify-icon icon="akar-icons:arrow-down"></iconify-icon>
                                  </span>
                                </p>
                                <div
                                  className={collapse ? "" : "collapse"}
                                  id="collapseExample"
                                >
                                  <div
                                    class="card card-body"
                                    style={{
                                      border: "none",
                                      padding: "0",
                                    }}
                                  >
                                    {edit === 1 && disable ? (
                                      <>
                                        {" "}
                                        <textarea
                                          style={{
                                            border: "none",
                                            minHeight: "100px",
                                          }}
                                          className="typing-container"
                                          value={data.address[0]}
                                          onChange={(event) => {
                                            let temp = data;
                                            temp.address[0] =
                                              event.target.value;
                                            setTxt(event.target.value);
                                            setData(temp);
                                            console.log(temp.address[0]);
                                          }}
                                        />
                                      </>
                                    ) : (
                                      add(data.address[0])
                                    )}
                                  </div>
                                </div>
                              </td>
                              <td>
                                {data.address[1] == ""
                                  ? "---"
                                  : data.address[1]}
                              </td>
                              <td >
                                <button
                                className="Pen"
                                  onClick={(e) => {
                                    e.preventDefault();
                                    setPrevTxt(data.address[0])
                                    setPrevScore(data.address[1])
                                    console.log(1);
                                    setEdit(1);
                                    setDisable(!disable);
                                    console.log(edit, 1);
                                  }}
                                >
                                  {edit === 1 && disable ? (
                                    <iconify-icon  icon="charm:tick"></iconify-icon>
                                  ) : (
                                    <iconify-icon  icon="uil:pen"></iconify-icon>
                                  )}
                                </button>
                              </td>
                            </tr>
                            <tr>
                              <td class="headcol">
                                <span class="dot" style={setBg(2)}></span>
                                Invoice  Date
                              </td>
                              <td>
                                <input
                                  style={{ border: "none" }}
                                  className="typing-container"
                                  value={data.date[0].text}
                                  onChange={(event) => {
                                    let temp = data;
                                    temp.date[0].text = event.target.value;
                                    setTxt(event.target.value);
                                    setData(temp);
                                    console.log(temp.date[0].text);
                                  }}
                                  readOnly={
                                    edit === 2 && disable ? false : true
                                  }
                                />
                              </td>
                              <td>
                                {data.date[1] == "" ? "---" : data.date[1]}
                              </td>
                              <td >
                                <button
                                className="Pen"
                                  onClick={(e) => {
                                    e.preventDefault();
                                    setPrevTxt(data.date[0].text)
                                    setPrevScore(data.date[1])
                                    setEdit(2);
                                    setDisable(!disable);
                                    console.log(disable);
                                  }}
                                >
                                  {edit === 2 && disable ? (
                                    <iconify-icon  icon="charm:tick"></iconify-icon>
                                  ) : (
                                    <iconify-icon  icon="uil:pen"></iconify-icon>
                                  )}
                                </button>
                              </td>
                            </tr>
                            <tr>
                              <td class="headcol">
                                <span class="dot" style={setBg(3)}></span>
                                Total
                              </td>
                              <td>
                                <input
                                  style={{ border: "none" }}
                                  className="typing-container"
                                  value={data.total[0].text}
                                  onChange={(event) => {
                                    let temp = data;
                                    temp.total[0].text = event.target.value;
                                    setTxt(event.target.value);
                                    setData(temp);
                                    console.log(temp.total[0].text);
                                  }}
                                  readOnly={
                                    edit === 3 && disable ? false : true
                                  }
                                />
                              </td>
                              <td>
                                {data.total[1] == "" ? "---" : data.total[1]}
                              </td>
                              <td >
                                <button
                                className="Pen"
                                  onClick={(e) => {
                                    e.preventDefault();
                                    setPrevTxt(data.total[0].text)
                                    setPrevScore(data.total[1])
                                    setEdit(3);
                                    setDisable(!disable);
                                    console.log(disable);
                                  }}
                                >
                                  {edit === 3 && disable ? (
                                    <iconify-icon  icon="charm:tick"></iconify-icon>
                                  ) : (
                                    <iconify-icon  icon="uil:pen"></iconify-icon>
                                  )}
                                </button>
                              </td>
                            </tr>
                            <tr>
                              <td class="headcol">
                                <span class="dot" style={setBg(3)}></span>
                                Category
                              </td>
                              <td>
                                {
                                  edit === 4 && disable ? <select
                                  style={{ border: "none" }}
                                  className="typing-container"
                                  value={data.category[0]}
                                  onChange={(event) => {
                                    let temp = data;
                                    temp.category[0] = event.target.value;
                                    setTxt(event.target.value);
                                    setData(temp);
                                    console.log(temp.category[0]);
                                  }}
                                 
                                >
                                  <option>Select a Category</option>
                                  <option value="Automotive">Automotive</option>
                                  <option value="Retail">Retail</option>
                                  <option value="Food">Food</option>
                                  <option value="Food">Healthcare</option>
                                  <option value="Electronics">Electronics</option>
                                </select> :  <input
                          style={{ border: "none" }}
                          className="typing-container"
                          value={data.category[0]}
                          readOnly={true}
                        />
                                }
                              
                               
                              </td>
                              <td>
                                {data.category[1] == "" ? "---" : data.category[1]}
                              </td>
                              <td >
                                <button
                                className="Pen"
                                  onClick={(e) => {
                                    e.preventDefault();

                                    setEdit(4);
                                    setDisable(!disable);
                                    console.log(disable);
                                  }}
                                >
                                  {edit === 4 && disable ? (
                                    <iconify-icon  icon="charm:tick"></iconify-icon>
                                  ) : (
                                    <iconify-icon  icon="uil:pen"></iconify-icon>
                                  )}
                                </button>
                              </td>
                              
                            </tr>
                            
                            <tr>
                              <td class="headcol">
                                <span class="dot" style={setBg(3)}></span>
                              Bill 
                              </td>
                              <td>
                             
                            
                            </td>
                              <td>
                               
                              </td>
                              <td >
                                <button
                                className="Pen"
                                  onClick={(e) => {
                                    e.preventDefault();

                                    setEdit(5);

                                    setBill(true)
                                    console.log(disable);
                                  }}
                                >
                                  {edit === 5 && disable ? (
                                    <iconify-icon  icon="charm:tick"></iconify-icon>
                                  ) : (
                                    <iconify-icon  icon="uil:pen"></iconify-icon>
                                  )}
                                </button>
                              </td>
                              </tr>
                              <br/>
                              
                          </tbody>
                        </table>
                        <table class="Tab " style={{marginTop:"0",marginLeft:"40px" ,     background: "#eee",
    borderBottom: "2px solid #adb5bd",boxShadow:"2px 2px 4px 4px #adb5bd"}} >
                          <thead>
                            {
                                data.headers.map((item)=>{
                                    return <th style={{borderTop:"none",padding:"11px 11px",fontSize:"unset"}}>{item}</th>
                                })
                            }
                           
                            
                          </thead>
                          <tbody>
                            {
                               data.values.map((it,index)=>{
                                if(it.length>data.headers.length){
                                  return ''
                                }
                                if(it.length<data.headers.length){
                                  let le=data.headers.length - it.length   
                                  return <tr>                 
                                    { data.headers.map((dat,ind)=> {
                                 
                                     if(ind<le){
                                      return  <td>  </td>
                                     }
                                     else{
                                      console.log(it)
                                      return  <td style={index!==data.values.length-1?{ padding: "15px",
                                        background: "#eee",
                                        borderBottom: "1px solid #fff"}:{padding: "15px",
                                        background: "rgb(30 30 30 / 85%)",
                                        color: "#ffffff",
                                        borderBottom: "1px solid #fff"}}> {it[ind-le]} </td>
                                     }
                                     }
                                                       
                                                        
                                   )}
                                      </tr> 
                                }
                               
                                return <tr>
                                 { it.map((dat,ind)=> {
                                 
                                  return  <td style={ind!==it.length-1?{ padding: "15px",
                                  background: "#eee",
                                  borderBottom: "1px solid #fff"}:{color:"black",
                                  borderBottom: "1px solid #fff",
                                    background: "#ddd"}}> {dat} </td>
                                 }
                                                   
                                                    
                               )}
                                </tr>
                               })
                            }
                            </tbody>
                            </table>
                      </div>
                      </div>
                      <div style={bill ? {} : {display:"none"}}>
                      <p className="TabHead">
                       Bill of M<span>aterials</span>
                      </p>
                      <button
                        className="Predictbtn"
                        onClick={(event) => {
                          event.preventDefault();
                          setBill(false)
                        }}
                      >
                        {" "}
                        <span> Save </span>
                      </button>
                      <div className="card-body Table" style={{overflowX : "scroll"}}>
                        <TableComponent headers={data.headers} values={data.values} data={data} setData={setData} />
                       
                        </div>
                     
                      </div>
                     
                      <br />
                      <br />
                    </>
                  )}
                   <br/>
                        <br/>
                        <br/>
                  <div className="TabFooter">
                   
                    
                    {" "}
                    {predicted ? (
                      <></>
                    ) : (

                      <button
                        className="Predictbtn"
                        onClick={(event) => {
                          event.preventDefault();
                          props.setFinalData(data);
                          props.setIndex(2);
                        }}
                      >
                        {" "}
                        <span> Save Changes </span>
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </fieldset>
        </>
      )}
    </>
  );
};

export default Page2Component;
