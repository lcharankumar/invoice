// import "./App.css";
import "../style/custom_css.css";
// icons
// import HomeIcon from '@mui/icons-material/Home';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';

import Button from "react-bootstrap/Button";
import Container from "react-bootstrap/Container";
import Form from "react-bootstrap/Form";
import Nav from "react-bootstrap/Nav";
import Navbar from "react-bootstrap/Navbar";
import NavDropdown from "react-bootstrap/NavDropdown";
import Offcanvas from "react-bootstrap/Offcanvas";

function Nav_testingComponent() {
  return (
    <>
      {/* false, 'sm', 'md', 'lg', 'xl', 'xxl' */}
      {[false].map((expand) => (
        <Navbar
          fixed="top"
          class="navbar_custom"
          key={expand}
          expand={expand}
          className="shadow-sm "
          style={{background:"white"}}
        >
          <Container fluid>
            <Navbar.Brand class=" p-2 ">
              {" "}
              <Row>
                <Col class=" custom_padding "><div class=" ">   </div></Col>
                <Col> {/* <iconify-icon icon="uil:home"></iconify-icon> */}
                  <span className="page_title_1">Invoice R</span>
                  <span className="page_title_2">ecognition</span>
                </Col>

              </Row>

            </Navbar.Brand>

            <Navbar.Toggle aria-controls={`offcanvasNavbar-expand-${expand}`} />
            <Navbar.Offcanvas
              id={`offcanvasNavbar-expand-${expand}`}
              aria-labelledby={`offcanvasNavbarLabel-expand-${expand}`}
              placement="end"
            >
              <Offcanvas.Header closeButton>
                <Offcanvas.Title id={`offcanvasNavbarLabel-expand-${expand}`}>
                  Offcanvas
                </Offcanvas.Title>
              </Offcanvas.Header>
              <Offcanvas.Body>
                <Nav className="justify-content-end flex-grow-1 pe-3">
                  <Nav.Link href="#action1">Home</Nav.Link>
                  <Nav.Link href="#action2">Link</Nav.Link>
                  <NavDropdown
                    title="Dropdown"
                    id={`offcanvasNavbarDropdown-expand-${expand}`}
                  >
                    <NavDropdown.Item href="#action3">Action</NavDropdown.Item>
                    <NavDropdown.Item href="#action4">
                      Another action
                    </NavDropdown.Item>
                    <NavDropdown.Divider />
                    <NavDropdown.Item href="#action5">
                      Something else here
                    </NavDropdown.Item>
                  </NavDropdown>
                </Nav>
                <Form className="d-flex">
                  <Form.Control
                    type="search"
                    placeholder="Search"
                    className="me-2"
                    aria-label="Search"
                  />
                  <Button variant="outline-success">Search</Button>
                </Form>
              </Offcanvas.Body>
            </Navbar.Offcanvas>
          </Container>
        </Navbar>
      ))}
    </>
  );
}

export default Nav_testingComponent;
