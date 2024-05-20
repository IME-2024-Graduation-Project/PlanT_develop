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
// import Dashboard from "views/Dashboard.js";
import Notifications from "views/Notifications.js";
import Icons from "views/Icons.js";
// import Typography from "views/Typography.js";
import TableList from "views/Tables.js";
import Maps from "views/Map.js";
import Pla1 from "views/Pla1";
import Pla2 from "views/Pla2";
import Map2 from "views/Map2";
import Map3 from "views/Map3";
import Login from "views/LoginGoogle";
// import UserPage from "views/User.js";
// import UpgradeToPro from "views/Upgrade.js";

var routes = [
  // {
  //   path: "/dashboard",
  //   name: "Dashboard",
  //   icon: "nc-icon nc-bank",
  //   component: <Dashboard />,
  //   layout: "/admin",
  // },
  {
    path: "/icons",
    name: "Input Page(1)",
    icon: "nc-icon nc-diamond",
    component: <Icons />,
    layout: "/admin",
  },
  {
    path: "/tables",
    name: "Input Page(2)",
    icon: "nc-icon nc-tile-56",
    component: <TableList />,
    layout: "/admin",
  },
  {
    path: "/notifications",
    name: "Input Page(3)",
    icon: "nc-icon nc-bell-55",
    component: <Notifications />,
    layout: "/admin",
  },
  {
    path: "/maps",
    name: "Output Page",
    icon: "nc-icon nc-pin-3",
    component: <Maps />,
    layout: "/admin",
  },
  {
    path: "/maps/map2",
    name: "Output Page",
    icon: "nc-icon nc-pin-3",
    component: <Map2 />,
    layout: "/admin",
    hidden: true,
  },
  {
    path: "/maps/map3",
    name: "Output Page",
    icon: "nc-icon nc-pin-3",
    component: <Map3 />,
    layout: "/admin",
    hidden: true,
  },
  {
    path: "/notifications/pla1",
    name: "Input Page(3)",
    icon: "nc-icon nc-bell-55",
    component: <Pla1 />,
    layout: "/admin",
    hidden: true,
  },
  {
    path: "/notifications/pla2",
    name: "Input Page(3)",
    icon: "nc-icon nc-bell-55",
    component: <Pla2 />,
    layout: "/admin",
    hidden: true,
  },
  {  
    path: "/login",
    name: "Login",
    icon: "nc-icon nc-bell-55",
    component: <Login />,
    layout: "/login",
    hidden: true,
  },
  // {
  //   path: "/pla1",
  //   name: "User Profile",
  //   icon: "nc-icon nc-single-02",
  //   component: <UserPage />,
  //   layout: "/admin",
  // },
  // {
  //   path: "/typography",
  //   name: "Typography",
  //   icon: "nc-icon nc-caps-small",
  //   component: <Typography />,
  //   layout: "/admin",
  // },
  // {
  //   pro: true,
  //   path: "/upgrade",
  //   name: "Upgrade to PRO",
  //   icon: "nc-icon nc-spaceship",
  //   component: <UpgradeToPro />,
  //   layout: "/admin",
  // },
];
export default routes;
