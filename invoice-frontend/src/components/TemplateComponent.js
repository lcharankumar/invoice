import React, { useEffect, useState } from "react";
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Button from 'react-bootstrap/Button';
import Card from 'react-bootstrap/Card';

import Nav_testingComponent from "./Nav_testingComponent";

import Page1Component from "./Page1Component";
import Page2Component from "./Page2Component";
import Page3Component from "./Page3Component";

const TempleteComponent = () => {
    const [index, setIndex] = useState(0);
    const [file, setFile] = useState(null);
    useEffect(() => {
        console.log(index);
    }, [index]);
    useEffect(() => {
        console.log(file);
    }, [file]);

    return (
        <>
            {" "}
            <Nav_testingComponent />
            <Container className=" container container-fluid ">
                <Row className="  ">
                  <span>progress bar</span>  
                </Row>
                <Row className="">
                <Col className=" col-6 ">
                        <Card>
                            <Card.Header>Featured</Card.Header>
                            <Card.Body>
                                <Card.Title>Special title treatment</Card.Title>
                                <Card.Text>
                                    With supporting text below as a natural lead-in to additional content.
                                </Card.Text>
                                <Button variant="primary">Go somewhere</Button>
                            </Card.Body>
                        </Card>
                    </Col>
                    <Col className=" col-6 ">
                        <Card>
                            <Card.Header>Featured</Card.Header>
                            <Card.Body>
                                <Card.Title>Special title treatment</Card.Title>
                                <Card.Text>
                                    With supporting text below as a natural lead-in to additional content.
                                </Card.Text>
                                <Button variant="primary">Go somewhere</Button>
                            </Card.Body>
                        </Card>
                    </Col>
                    
                </Row>

            </Container>
        </>
    );
};

export default TempleteComponent;
