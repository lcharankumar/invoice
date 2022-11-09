import React, { useEffect, useState } from "react";
import img from "../img/img2.png";
import { TestData3 } from "../data";
const Page3Component = (props) => {
  var today = new Date();
var dd = String(today.getDate()).padStart(2, '0');
var mm = String(today.getMonth() + 1).padStart(2, '0'); //January is 0!
var yyyy = today.getFullYear();

today = mm + '/' + dd + '/' + yyyy;
  const [data, setData] = useState({
    address: ["", ""],
    date: [{text:""}, ""],
    invono: [{text:""}, ""],

    total: [{text:""}, ""],
    category:[
      "Retail",
      "NA"
    ],
    headers:[],
    values:[]
  });
  const [collapse, setCollapse] = useState(false);
  useEffect(() => {
    if (props.finalData) {
      setData(props.finalData);
    }
    console.log(props.finalData);
  }, [props.finalData]);
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
      <fieldset>
        <div class="container" style={{ minHeight: "731px" }}>
          <div class="row">
            <div class="col">
              <div class="card">
                
                <div class="card-body">
                  
                  {/* <img
                    class="card-img-top"
                    src={props.final}
                    alt="Card image cap"
                  /> */}
                   <img
                    class="card-img-top"
                    src={props.org}
                    alt="Card image cap"
                  />
                </div>
              </div>
            </div>
            <div class="Tabcol col card" style={{ border: "none" }}>
              <p className="TabHead">
                Predicted L<span>ables</span>
              </p>
              <div className="card-body">
                <table class="Tab ">
                  <thead>
                    <th>Label</th>
                    <th>Text</th>
                  </thead>
                  <tbody>
                  <tr>
                      <td>
                        <span
                          class="dot"
                          style={{ background: "orange" }}
                        ></span>
                       User
                      </td>
                      <td>
                        <input
                          style={{ border: "none" }}
                          className="typing-container"
                          value="Admin"
                          readOnly={true}
                        />
                      </td>
                    </tr>
                    <tr>
                      <td>
                        <span
                          class="dot"
                          style={{ background: "orange" }}
                        ></span>
                        Submitted date
                      </td>
                      <td>
                        <input
                          style={{ border: "none" }}
                          className="typing-container"
                          value={today}
                          readOnly={true}
                        />
                      </td>
                    </tr>
                    <tr>
                      <td>
                        <span
                          class="dot"
                          style={{ background: "orange" }}
                        ></span>
                        Category
                      </td>
                      <td>
                        <input
                          style={{ border: "none" }}
                          className="typing-container"
                          value={data.category[0]}
                          readOnly={true}
                        />
                      </td>
                    </tr>
                    <tr>
                      <td>
                        <span
                          class="dot"
                          style={{ background: "orange" }}
                        ></span>
                        Invoice No
                      </td>
                      <td>
                        <input
                          style={{ border: "none" }}
                          className="typing-container"
                          value={data.invono[0].text}
                          readOnly={true}
                        />
                      </td>
                    </tr>
                    <tr>
                      <td>
                        <span
                          class="dot"
                          style={{ background: "orange" }}
                        ></span>
                       Invoice Date
                      </td>
                      <td>
                        <input
                          style={{ border: "none" }}
                          className="typing-container"
                          value={data.date[0].text}
                          readOnly={true}
                        />
                      </td>
                    </tr>
                    <tr>
                      <td className="Add">
                        <span
                          class="dot"
                          style={{ background: "orange" }}
                        ></span>
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
                            : data.address[0].substring(0, 6) + "  ...  "}

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
                            {add(data.address[0])}
                          </div>
                        </div>
                      </td>
                    </tr>
                    
                    <tr>
                      <td>
                        <span
                          class="dot"
                          style={{ background: "orange" }}
                        ></span>
                        Total
                      </td>
                      <td>
                        <input
                          style={{ border: "none" }}
                          className="typing-container"
                          value={data.total[0].text}
                          readOnly={true}
                        />
                      </td>
                    </tr>
                    <tr>
                      <td>
                        <span
                          class="dot"
                          style={{ background: "orange" }}
                        ></span>
                       Bill of materials
                      </td>
                     
                    </tr>
                    
                    <br/>
                  </tbody>
                </table>
                <div style={{overflowX : "scroll"}}>
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
                            <br/>
                            </div>
              </div>
              <br/>
              <br/>
              <br/>

              <div className="TabFooter">
                {" "}
                <button
                  className="Predictbtn"
                  style={{ background: "#ffffff",
  border: "1px solid #f87115",
  borderRadius: "15px",
  padding: "8px 25px",
  textTransform: "uppercase",
  fontSize: "13px",
  fontWeight: "500",
  letterSpacing: "1px",
  color: "#f87115",
  marginRight:"40px"
  }}
                  onClick={(event) => {
                    event.preventDefault();

                    props.setIndex(1);
                  }}
                >
                  {" "}
                  <span> Back </span>
                </button>
                <button
                  className="Predictbtn"
                  onClick={(event) => {
                    event.preventDefault();

                    props.setIndex(0);
                  }}
                >
                  {" "}
                  <span> Review & Submit </span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </fieldset>
    </>
  );
};

export default Page3Component;
