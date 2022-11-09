import React, { useState, useEffect } from "react";
import ReactCrop, {
  centerCrop,
  makeAspectCrop,
  Crop,
  PixelCrop,
} from 'react-image-crop'

import "react-image-crop/dist/ReactCrop.css";
import img1 from "../img/img1.jpg";
import { Link } from "react-router-dom";
function ImageCropper(props) {
  console.log("img comp", props);
  const myRef = React.createRef();
  const onImageCropped = (croppedImage) => setCroppedImage(croppedImage);
  const [imageToCrop, setImageToCrop] = useState(props.img);
  const [croppedImage, setCroppedImage] = useState(myRef);
  const pixelRatio = window.devicePixelRatio;


  let currh = props.currh;
  let setCurrh = props.setCurrh;

  let currw = props.currw;
  let setCurrw = props.setCurrw;

  const [cropConfig, setCropConfig] = useState(
    // default crop config
    {
      unit: "%",
      width: 90,
      aspect: 16 / 9,
    }
  );

  const [imageRef, setImageRef] = useState(myRef);

  async function cropImage(crop) {
    console.log(myRef.current);
    setCurrh(myRef.current.clientHeight);
    setCurrw(myRef.current.clientWidth);
    if (imageRef && crop.width && crop.height) {
      const croppedImage = await getCroppedImage(
        myRef.current,
        crop,
        "croppedImage.jpeg" // destination filename
      );

      // calling the props function to expose
      // croppedImage to the parent component
      setCroppedImage(croppedImage);
      props.setCrop(croppedImage);
    }
  }

  function getCroppedImage(sourceImage, cropConfig, fileName) {
    console.log("HIII");
    // creating the cropped image from the source image
    const canvas = document.createElement("canvas");
    const scaleX = sourceImage.naturalWidth / sourceImage.width;
    const scaleY = sourceImage.naturalHeight / sourceImage.height;
    canvas.width = cropConfig.width * scaleX * pixelRatio;
    canvas.height = cropConfig.height * scaleY * pixelRatio;
    const ctx = canvas.getContext("2d");
    
    console.log(
      cropConfig.x,
      cropConfig.y,
      cropConfig.width,
      cropConfig.height
    );
    let temp = {
      x: cropConfig.x,
      y: cropConfig.y,
      w: cropConfig.width,
      h: cropConfig.height,
    };
    props.setCoordinates(temp);
    console.log(temp);
    
    ctx.imageSmoothingQuality = 'high'
    
  ctx.scale(pixelRatio, pixelRatio)
    ctx.drawImage(
      sourceImage,
      cropConfig.x * scaleX,
      cropConfig.y * scaleY,
      cropConfig.width * scaleX,
      cropConfig.height * scaleY,
      0,
      0,
      cropConfig.width,
      cropConfig.height
    );
   
    // return new Promise((resolve, reject) => {
    //   canvas.toBlob((blob) => {
    //     // returning an error
    //     if (!blob) {
    //       reject(new Error("Canvas is empty"));
    //       return;
    //     }

    //     blob.name = fileName;
    //     // creating a Object URL representing the Blob object given
    //     const croppedImageUrl = window.URL.createObjectURL(blob);
    //     console.log(croppedImage);
    //     resolve(croppedImageUrl);
    //   }, "image/jpeg");
    // });
    return canvas.toDataURL("image/png",1)
   
  }

  return (
    <div className="app">
      <div>
        <ReactCrop
          crop={cropConfig}
          ruleOfThirds
          onImageLoaded={(imageRef) => {
            setImageRef(myRef);
            console.log(imageRef);
          }}
          onComplete={(cropConfig) => {
            cropImage(cropConfig);
          }}
          onChange={(cropConfig) => {
            setCropConfig(cropConfig);
          }}
        >
          <img ref={myRef} src={imageToCrop} alt="" />
        </ReactCrop>
      </div>
      
      
    </div>
  );
}

ImageCropper.defaultProps = {
  onImageCropped: () => {},
};

export default ImageCropper;
