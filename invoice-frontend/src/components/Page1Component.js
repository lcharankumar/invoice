import React, { useState,useEffect } from "react";
import ZoomComponent from "./ZoomComponent";
const Page1Component = (props) => {
  const [files, setFiles] = useState([]);
  const [pdf, setPdf] = useState(false);
   const [pdfFile,setPdfFile]= useState(null)
  
  let file = props.file;
  let setFile = props.setFile;
 
   
  function overrideEventDefaults(event) {
    event.preventDefault();
    event.stopPropagation();
  }
  function handleDragAndDropFiles(event) {
    overrideEventDefaults(event);
    if (!event.dataTransfer) return;
    handleFiles(event.dataTransfer.files);
  }
  const handleFiles = (fileList) => {
    if (fileList) {
      let files = Array.from(fileList);
      props.setName(files[0].name.slice(-12))
      setPdfFile(files[0])
      console.log(files[0].type)
      if(files[0].type=="application/pdf"){
        setPdf(true)
        
       }
       else{
        setFiles(files);
        
        props.setFile(files[0]);
        const reader = new FileReader();
        reader.addEventListener("load", () =>
          props.setImg(reader.result.toString() || "")
        );
        reader.readAsDataURL(files[0]);
        props.setIndex(1);
       }
      
      
      
      console.log(props.img);
    }
  };

  return (
    <>
      <fieldset className="Upload">
        

        <form
          id="file-upload-form"
          class="uploader"
          onDrop={handleDragAndDropFiles}
          onDragEnter={overrideEventDefaults}
          onDragLeave={overrideEventDefaults}
          onDragOver={overrideEventDefaults}
        >
          <input
            id="file-upload"
            type="file"
            name="fileUpload"
            onChange={(e) => {
              if (e.target.files) {
                e.preventDefault();
                console.log(e.target.files[0])
                props.setName(e.target.files[0].name.slice(-12))
               
                setPdfFile(e.target.files[0])
                if(e.target.files[0].type=="application/pdf"){
                 setPdf(true)
                
                 
                }
                
               else{
                props.setFile(e.target.files[0]);
                console.log(e.target.files);
                props.setName(e.target.files[0].name.slice(-12))
                const reader = new FileReader();
                reader.addEventListener("load", () =>{
                  props.setImg(reader.result.toString() || "");
                  props.setOrg(reader.result.toString() || "");
                }
                );
                reader.readAsDataURL(e.target.files[0]);
                console.log(props.img);
                props.setIndex(1);
               }
                
              }
            }}
          />
          <label for="file-upload" id="file-drag">
            <img id="file-image" src="#" alt="Preview" class="hidden" />
            <div id="start">
            <p class="lead">File format supported : pdf, jpg, png</p>
              <div style={{ color: "#000000", fontWeight: "600" }}>
                Select a file or drag here
              </div>
              <div id="notimage" class="hidden">
                Please select an image
              </div>
              <span id="file-upload-btn" class="btn btn-primary">
                Select a file
              </span>
            </div>
            <div id="response" class="hidden">
              <div id="messages"></div>
              <progress class="progress" id="file-progress" value="0">
                <span>0</span>%
              </progress>
            </div>
          </label>
        </form>
        {
          pdf? <ZoomComponent setOrg={props.setOrg} setImg={props.setImg} setFile={props.setFile} setPdf={setPdf} file={pdfFile} setIndex={props.setIndex}/> :''
        }
      </fieldset>
    </>
  );
};

export default Page1Component;
