import React, { useRef, useEffect, useState,useLayoutEffect } from "react";
import img1 from "../img/img1.png";
import img2 from "../img/original.png";
import Button from 'react-bootstrap/Button';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import Modal from 'react-bootstrap/Modal';
import Row from 'react-bootstrap/Row';


const RectangleComponent = (props) => {
  console.log("rect comp", props);
  const [modalShow, setModalShow] = useState(true);
  const pixelRatio = window.devicePixelRatio || 1
  const canvas = useRef();
  const targetRef = useRef();
  let ctx = null;
  const[imgh,setImgh]=useState(0)
  const[imgw,setImgw]=useState(0)

  const [myImage, setMyImage] = useState(props.img);
  
  let final = props.final;
  let setMyFinal = props.setMyFinal;
  let currh = props.currh;
  let setCurrh = props.setCurrh;

  let currw = props.currw;
  let setCurrw = props.setCurrw;

  // initialize the canvas context
  useLayoutEffect(() => {
    if (targetRef.current) {
         console.log("Before",targetRef.current.offsetWidth,targetRef.current.offsetHeight)
        
       setImgh(targetRef.current.offsetHeight)
       setImgw(targetRef.current.offsetWidth)
       //console.log(currh,currw)

    }
  }, []);
  useEffect(() => {
    console.log(props.img);
    // dynamically assign the width and height to canvas

    const canvasEle = canvas.current;
    canvasEle.width = canvasEle.clientWidth;
    canvasEle.height = canvasEle.clientHeight;

    // get context of the canvas
    ctx = canvasEle.getContext("2d");
    const image = new Image();
    image.src = myImage;
    //image.src= img1
    // image.width=currw
    // image.height=currh
    
    console.log("After",image)

    console.log(currh, currw);
    //ctx.drawImage(image, 0, 0);

    const timer = setTimeout(() => {
    
      
      //ctx.scale(currw * pixelRatio, currh*pixelRatio)
      console.log(imgw,imgh)
      ctx.drawImage(image,0,0,currw,currh);
      

      const r1Info = props.coordinates;
      const r1Style = { borderColor: "red", borderWidth: 2 };
      drawRect(r1Info, r1Style);
      props.setImg(canvasEle.toDataURL("image/png",1));
      setMyImage(canvasEle.toDataURL("image/png",1));
      setMyFinal(canvasEle.toDataURL("image/png",1));
    }, 1000);
  }, []);

  const drawRect = (info, style = {}) => {
    console.log(info);
    const { x, y, w, h } = info;
    const { borderColor = "black", borderWidth = 1 } = style;

    ctx.beginPath();
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = borderWidth;
    ctx.rect(x, y, w, h);
    ctx.stroke();
    ctx.fillStyle = "red";
    ctx.imageSmoothingEnabled = false;
    ctx.fillText(props.txt, x, y-2 );
    setMyFinal(canvas.current.toDataURL("image/png",1));
  };
  const canvasToImg = (_) => {
    let tagA = document.createElement("a");
    document.body.appendChild(tagA);
    tagA.href = canvas.current.toDataURL();
    tagA.download = "canvas-image.png";
    tagA.click();
    document.body.removeChild(tagA);
    //  myImage = canvas.toDataURL("image/png");
  };

  return (
   
      
       <Modal style={{minWidth:"100vw"}} show={modalShow} onHide={() => {setModalShow(false);props.setRect(false)}} size="lg" aria-labelledby="contained-modal-title-vcenter">
      <Modal.Header closeButton>
        <Modal.Title id="contained-modal-title-vcenter" style={{    fontStyle: "italic",
    fontSize: "18px",    fontWeight: "bold"}}>
         Successfully Extracted text for {props.label} as {props.extract}
        </Modal.Title>
      </Modal.Header>
      <Modal.Body className="show-grid">
        <Container style={{width:"fit-content !important"}}>
        
            <canvas ref={canvas}  width={currw}
  height={currh } style={{ height: currh , width: currw  }}>
        
      </canvas>
      
      

                  </Container>
      </Modal.Body>
      <Modal.Footer>
        <Button style={{ color: "#ffffff",
  border: "1px solid #f87115",
  borderRadius: "15px",
  padding: "8px 25px",
  textTransform: "uppercase",
  fontSize: "13px",
  fontWeight: "500",
  letterSpacing: "1px",
  background: "#f87115"
  }}  onClick={()=>{props.setCancel(true);setModalShow(false);props.setRect(false)}}>Save changes</Button>
        <Button style={{ background: "#ffffff",
  border: "1px solid #f87115",
  borderRadius: "15px",
  padding: "8px 25px",
  textTransform: "uppercase",
  fontSize: "13px",
  fontWeight: "500",
  letterSpacing: "1px",
  color: "#f87115"
  }} onClick={()=>{props.revert();setModalShow(false);props.setRect(false)}}>Discard changes</Button>
      </Modal.Footer>
     
    </Modal>
     
     
   
  );
};

export default RectangleComponent;
