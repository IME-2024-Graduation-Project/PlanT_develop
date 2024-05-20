import React, { useState } from "react";
import NotificationAlert from "react-notification-alert";
import {
  Button,
  Card,
  CardHeader,
  CardBody,
  CardTitle,
  Row,
  Col,
  Table,
  Nav,
  NavItem,
  NavLink,
  UncontrolledDropdown, 
  DropdownToggle, 
  DropdownMenu, 
  DropdownItem
} from "reactstrap";
import logo from '../plant.png'

function Notifications() {
  const notificationAlert = React.useRef();
  const [selectedTag, setSelectedTag] = useState('태그를 선택하세요');
  const [selectedbutton, setSelectedbutton] = useState('');
  const [selectedtable, setSelectedtable] = useState('');

  const notify = (place) => {
    var color = Math.floor(Math.random() * 5 + 1);
    var type;
    switch (color) {
      case 1:
        type = "primary";
        break;
      case 2:
        type = "success";
        break;
      case 3:
        type = "danger";
        break;
      case 4:
        type = "warning";
        break;
      case 5:
        type = "info";
        break;
      default:
        break;
    }
    var options = {};
    options = {
      place: place,
      message: (
        <div>
          <div>
            Welcome to <b>Paper Dashboard React</b> - a beautiful freebie for
            every web developer.
          </div>
        </div>
      ),
      type: type,
      icon: "nc-icon nc-bell-55",
      autoDismiss: 7,
    };
    notificationAlert.current.notificationAlert(options);
  };

  const handleTagSelect = (tag) => {
    setSelectedTag(tag); // 선택된 태그 업데이트
  };

  const handleButtonSelect = (button) => {
    setSelectedbutton(button); // 선택된 태그 업데이트
  };

  const handletableSelect = (table) => {
    setSelectedtable(table); // 선택된 태그 업데이트
  };


  return (
    <>
      <div className="content">
        <NotificationAlert ref={notificationAlert} />
        <Row>
          <Col md="12">
            <Card>
              <CardHeader>
                <CardTitle tag="h5">Input for Trip(3/3)</CardTitle>
                <Nav tabs>
                  <NavItem>
                        <NavLink href="#" active>1일차</NavLink>
                    </NavItem>
                    <NavItem>
                        <NavLink href="/admin/notifications/pla1">2일차</NavLink>
                    </NavItem>
                    <NavItem>
                        <NavLink disabled href="#">3일차(이동일정)</NavLink>
                    </NavItem>
                    <NavItem>
                        <NavLink href="#">4일차</NavLink>
                    </NavItem>
                </Nav>
              </CardHeader>
              <CardBody>
                <Row>
                  <Col md="6">
                    <Card className="card-plain">
                      <CardHeader>
                        <CardTitle tag="h5">Additional Schedule</CardTitle>
                      </CardHeader>
                      <CardBody>
                        <UncontrolledDropdown group>
                          <DropdownToggle caret color="info">
                            {selectedTag}
                          </DropdownToggle>
                          <DropdownMenu>
                            <DropdownItem onClick={() => handleTagSelect("쇼핑")}>쇼핑</DropdownItem>
                            <DropdownItem onClick={() => handleTagSelect("식사")}>식사</DropdownItem>
                            <DropdownItem onClick={() => handleTagSelect("관광")}>관광</DropdownItem>
                            <DropdownItem onClick={() => handleTagSelect("카페")}>카페</DropdownItem>
                            <DropdownItem onClick={() => handleTagSelect("이동")}>이동</DropdownItem>
                          </DropdownMenu>
                        </UncontrolledDropdown>
                        <br />
                        <hr />
                        <div className="text-center">
                          <Button color="warning" fade={false}>
                            <span>그랜트 하얏트 서울(숙박)</span>
                          </Button>
                          <br />
                          <Button color="secondary" onClick={() => handleButtonSelect('+')}>+</Button>
                          <br />
                          {selectedtable === 'on' && (
                            <div>  
                              <Button color="success" fade={false}>
                                <span>풀리너마이트(식사)</span>
                              </Button>
                              <br />
                              <Button color="secondary">+</Button>
                              <br />
                            </div>
                          )}
                          <Button color="success" fade={false}>
                            <span>평화의 공원(관광)</span>
                          </Button>
                          <br />
                          <Button color="secondary">+</Button>
                          <br />
                          <Button
                            color="danger"
                            fade={false}
                          >
                            <span>서서울호수공원(관광)</span>
                          </Button>
                          <br />
                          <Button color="secondary">+</Button>
                          <br />
                          <Button
                            color="warning"
                            fade={false}
                          >
                            <span>그랜트 하얏트 서울(숙박)</span>
                          </Button>
                        </div>
                      </CardBody>
                    </Card>
                  </Col>
                  <Col md="6">
                    <Card className="card-plain">
                      <CardHeader>
                        <CardTitle tag="h5">Possible Places</CardTitle>
                      </CardHeader>
                      <CardBody>
                        <br />
                        <br />
                        <Table responsive>
                          <thead className="text-primary">
                            <tr>
                              <th>Name</th>
                              <th>Tag</th>
                              <th>City</th>
                              <th>Eco</th>
                            </tr>
                          </thead>
                          {selectedbutton === '+' && (
                          <tbody>
                            <tr>
                              <td>뉴욕비앤씨 신촌본점</td>
                              <td>식사</td>
                              <td>서울</td>
                              <td><img src={logo} alt="react-logo" width="30" height="30" /></td>
                            </tr>
                            <tr>
                              <td>점점점점점점</td>
                              <td>식사</td>
                              <td>서울</td>
                              <td><img src={logo} alt="react-logo" width="30" height="30" /></td>
                            </tr>
                            <tr>
                              <td>아민 연남</td>
                              <td>식사</td>
                              <td>서울</td>
                              <td><img src={logo} alt="react-logo" width="30" height="30" /></td>
                            </tr>
                            <tr>
                              <td>마이 리틀 마운틴</td>
                              <td>식사</td>
                              <td>서울</td>
                              <td><img src={logo} alt="react-logo" width="30" height="30" /></td>
                            </tr>
                            
                            <tr onClick={() => handletableSelect('on')}>
                              <td>풀리너마이트 홍대</td>
                              <td>식사</td>
                              <td>서울</td>
                              <td></td>
                            </tr>
                            <tr>
                              <td>독타운피자</td>
                              <td>식사</td>
                              <td>서울</td>
                              <td></td>
                            </tr>
                            <tr>
                              <td>저스트텐동 연남본점</td>
                              <td>식사</td>
                              <td>서울</td>
                              <td></td>
                            </tr>
                            <tr>
                              <td>키친31</td>
                              <td>식사</td>
                              <td>서울</td>
                              <td></td>
                            </tr>
                           </tbody>  
                          )}  
                          {!selectedbutton && (
                          <tbody> 
                           <tr>
                            <td></td>
                            <td></td>
                            <td></td>
                            <td></td>
                           </tr> 
                          </tbody>
                          )}
                          

                        </Table>
                        <br />
                        <br />
                        <br />
                        <br />
                        {selectedtable === 'on' && (
                        <div className="text-center">
                          <p>11h(default) - 2h * 3(일정 개수, 숙소 제외) = 5h</p>
                          <h5> 현재 가용시간 : 5h </h5>
                        </div>
                        )}
                        {!selectedtable && (
                        <div className="text-center">
                          <p>11h(default) - 2h * 2(일정 개수, 숙소 제외) = 7h</p>
                          <h5> 현재 가용시간 : 7h </h5>
                        </div>
                        )}
                        <br />
                      </CardBody>
                    </Card>
                  </Col>
                </Row>
              </CardBody>
              <br />
              <br />
              <br />
              <div>
                <Col className="ml-auto mr-auto" lg="8">
                  <Row>
                    <Col md="4">
                      <Button
                        block
                        className="btn-round"
                        color="primary"
                        href="http://localhost:3000/admin/icons">
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
                        Next
                      </Button>
                    </Col>
                  </Row>
                </Col>
              </div>
            </Card>
          </Col>
        </Row>

      </div>
    </>
  );
}

export default Notifications;
