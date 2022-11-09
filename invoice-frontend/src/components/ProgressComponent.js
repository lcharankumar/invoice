import React, { useEffect, useState } from "react";

import Nav_testingComponent from "./Nav_testingComponent";
// import "../style/custom_css.css";
import Page1Component from "./Page1Component";
import Page2Component from "./Page2Component";
import Page3Component from "./Page3Component";

const ProgressComponent = (props) => {
  const [index, setIndex] = useState(0);
  const [file, setFile] = useState(null);
  const [finalData, setFinalData] = useState(null);
  const [final, setMyFinal] = useState(null);
  const [crop, setCrop] = useState(null);
  const [org, setOrg] = useState(null);
  const [name,setName] = useState(null)
  useEffect(() => {
    console.log(index);
    console.log(file);
  }, [index]);
  useEffect(() => {
    console.log(file);
  }, [file]);

  return (
    <>
      {" "}
      <section class="multi_step_form">
        <form id="msform">
          {/* <div className="tittle"> */}
          <Nav_testingComponent />
          {/* </div> */}

          <ul id="progressbar" class=" p-4 " style={{marginTop:"70px"}}>
            <li
              className={index >= 0 ? " active" : ""}
              style={index == 0 ? { color: "red" } : {}}
            >
              Upload file
            </li>
            <li
              className={index >= 1 ? " active" : ""}
              style={index == 1 ? { color: "red" } : {}}
            >
              Verify & Edit
            </li>
            <li
              className={index === 2 ? " active" : ""}
              style={index == 2 ? { color: "red" } : {}}
            >
              Review & Submit
            </li>
          </ul>
          <div
            id="carouselExampleControls"
            class="carousel slide"
            data-ride="carousel"
          >
            <div class="carousel-inner" style={{ overflow: "unset" }}>
              <div
                className={
                  index === 0 ? "carousel-item active" : "carousel-item"
                }
              >
                <Page1Component
                  index={index}
                  setIndex={setIndex}
                  setFile={setFile}
                  img={props.img}
                  setImg={props.setImg}
                  setName={setName}
                  setOrg={setOrg}
                />
              </div>
              <div
                className={
                  index === 1 ? "carousel-item active" : "carousel-item"
                }
              >
                <Page2Component
                  index={index}
                  setIndex={setIndex}
                  setFinalData={setFinalData}
                  coordinates={props.coordinates}
                  setCoordinates={props.setCoordinates}
                  img={props.img}
                  setImg={props.setImg}
                  file={file}
                  setMyFinal={setMyFinal}
                  setCrop={setCrop}
                  crop={crop}
                  final={final}
                  name={name}

                />
              </div>
              <div
                className={
                  index === 2 ? "carousel-item active" : "carousel-item"
                }
              >
                <Page3Component
                  index={index}
                  setIndex={setIndex}
                  finalData={finalData}
                  final={final}
                  org={org}
                />
              </div>
            </div>
          </div>
        </form>
      </section>
    </>
  );
};

export default ProgressComponent;
