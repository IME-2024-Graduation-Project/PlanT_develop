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
import { Card, CardHeader, CardBody, Row, Col, CardTitle, Nav, Button, NavItem, NavLink } from "reactstrap";
import MapWrapper from "./Mapwraper";

function Map3() {

  const placeIds = [1315, 1261, 1077, 1264, 1315];

  return (
    
      <div className="content">
        <Row>
          <Col md="12">
            <Card>
              <CardHeader>
                <CardTitle tag="h5">Output for your Trip</CardTitle>
                  <Nav tabs>
                      <NavItem>
                          <NavLink href="/admin/maps">1일차</NavLink>
                      </NavItem>
                      <NavItem>
                          <NavLink href="/admin/maps/map2">2일차</NavLink>
                      </NavItem>
                      <NavItem>
                          <NavLink disabled href="#">3일차(이동일정)</NavLink>
                      </NavItem>
                      <NavItem>
                          <NavLink active>4일차</NavLink>
                      </NavItem>
                  </Nav>
              </CardHeader>
              <CardBody>
                <div
                  id="map"
                  className="map"
                  style={{ position: "relative", overflow: "hidden" }}
                >
                  <MapWrapper placeIds={placeIds} />
                </div>
                  <br/>
                  <h5> 선택된 친환경 장소 합계 : 6개</h5><p>6 / 12 * 100 * 10%(대중교통 보너스) = 55point 적립 예정</p>
                  <br/>
                  <br/>
                      <div> 
                                
                        <Col className="ml-auto mr-auto" lg="8">
                          <Row>
                            <Col md="4">
                              <Button 
                                block
                                className="btn-round" 
                                color="primary"
                                href="http://localhost:3000/admin/notifications">
                                Prev
                              </Button>
                            </Col>
                            <Col md="4">
                            </Col>
                            <Col md="4">
                              <Button 
                                block
                                className="btn-round" 
                                color="primary" 
                                type="submit"
                                href="http://localhost:3000/admin/maps">
                                경로 상세보기
                              </Button>
                            </Col>
                          </Row>
                        </Col>  
                      </div> 
              </CardBody>
            </Card>
          </Col>
        </Row>
      </div>
  );
}

export default Map3;
