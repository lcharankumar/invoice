import { imageListClasses } from '@mui/material';
import React,{useState,useEffect} from 'react'
import Carousel from 'react-bootstrap/Carousel';

import Button from 'react-bootstrap/Button';
import Card from 'react-bootstrap/Card';
import Modal from 'react-bootstrap/Modal';
const pdfjs = require("pdfjs-dist/webpack");
const ZoomComponent = (props) => {
  const [files,setFile]=useState(null)
  const [index, setIndex] = useState(0);
  const [imglist, setImglist] = useState([]);
  const [loaded, setLoaded] = useState(false);
  const [show, setShow] = useState(false);

  const handleClose = () => setShow(false);
  const handleShow = () => setShow(true);
 
  useEffect(() => {
  convertPdfToImages(props.file)
  }, [])
  

  const handleSelect = (selectedIndex, e) => {
    setIndex(selectedIndex);
    console.log(selectedIndex)
    console.log(imglist[selectedIndex])
   
  };
  const readFileData = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        resolve(e.target.result);
      };
      reader.onerror = (err) => {
        reject(err);
      };
      reader.readAsDataURL(file);
    });
   
  };
  
  //param: file -> the input file (e.g. event.target.files[0])
  //return: images -> an array of images encoded in base64 
  const convertPdfToImages = async (file) => {
    
   
    const images = [];
    const data = await readFileData(file);
    const pdf = await pdfjs.getDocument(data).promise;
    const canvas = document.createElement("canvas");
    for (let i = 0; i < pdf.numPages; i++) {
      const page = await pdf.getPage(i + 1);
      const viewport = page.getViewport({ scale: 1 });
      const context = canvas.getContext("2d");
      canvas.width = viewport.width ;
canvas.height = viewport.height;
context.imageSmoothingQuality = 'high'
      await page.render({ canvasContext: context, viewport: viewport }).promise;
      images.push(canvas.toDataURL("image/png",1));
    }
    canvas.remove();
   console.log(images)
   setImglist(images)
   setLoaded(true)
   handleShow()
  }
  return (
    
    
    <Modal show={show} onHide={handleClose}  size="lg"
      aria-labelledby="contained-modal-title-vcenter"
      className='modal-dialog-centered' style={{minWidth:"100vw"}}>
    <Modal.Header closeButton >
      <Modal.Title style={{fontStyle:"italic"}}>Select the page you want to predict</Modal.Title>
      
    </Modal.Header>
    <Modal.Body> 
  <Carousel interval={null} slide={false} activeIndex={index} onSelect={handleSelect} style={loaded?{}:{display:"none"}} variant="dark">
  {
        imglist.map((item)=>{
          return  <Carousel.Item >
      <img
          className="d-block w-100"
          src={item}
          alt="Second slide"
        />
         
        </Carousel.Item>
        
        })
       }
      
    </Carousel>
 
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
  }}  onClick={()=>{
        console.log(index)
        props.setImg(imglist[index])
        props.setOrg(imglist[index])
        var arr = imglist[index].split(','), mime = arr[0].match(/:(.*?);/)[1],
        bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
        while(n--){
        u8arr[n] = bstr.charCodeAt(n);
        }
      let imageFile1= new File([u8arr], "cropped.png", {type:"image/png"});
      
        props.setFile(imageFile1)
        props.setIndex(1)
        setShow(false)
      }}>
        Select 
      </Button>
</Modal.Footer>
   
  </Modal>
 
   
  )
}

export default ZoomComponent