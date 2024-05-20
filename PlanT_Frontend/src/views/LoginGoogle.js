/*!

=========================================================
* Paper Dashboard React - v1.3.2
=========================================================

* Product Page: https://www.creative-tim.com/product/paper-dashboard-react
* Copyright 2023 Creative Tim (https://www.creative-tim.com)

* Licensed under MIT (https://github.com/creativetimofficial/paper-dashboard-react/blob/main/LICENSE.md)

* Coded by Creative Tim

=========================================================

* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

*/
import React from "react";
// reactstrap components
import 
{ Card,
  Button,
  CardBody,
  CardFooter,
  Row,

  Col,} from "reactstrap";
import GoogleLoginButton from "./GoogleLoginButton";

function Login() {



  return (
    
      <div className="content">
        <Row>
          <Col md="12">
          <Card className="card-user">
              <div className="image">
                <img alt="..." src={require("assets/img/damir-bosnjak.jpg")} />
              </div>
              <CardBody>
                <div className="author">
                    <img
                      alt="..."
                      className="avatar border-gray"
                      src={require("assets/img/traveler.png")}
                    />
                    <h1 className="title">PlanT</h1>
                </div>
                <h5 className="description text-center">
                  Preference <br />
                  + <br />Trend <br />
                  + <br />AI
                  <br /> = 
                  <br />PlanT's Plans
                </h5>
              </CardBody>
              <CardFooter>
                <hr />
                <div className="button-container">
              
                <h4 className="text-center">Social Login</h4>
                <br/>
                  <Row className="ml-auto mr-auto" lg="1">
                  <div>
                    <GoogleLoginButton />
                  </div>
                  </Row>
                  <br/>
                  <br/>
                  <br/>
                </div>
              </CardFooter>
            </Card>
          </Col>
        </Row>
      </div>
  );
}

export default Login;
