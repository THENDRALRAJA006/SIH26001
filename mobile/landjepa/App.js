/**
 * LAND-JEPA Mobile — Root Navigation
 * Bottom tab navigator: Zones | Alerts | Report
 */
import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { Text } from "react-native";
import HomeScreen      from "./src/app/HomeScreen";
import ZoneDetailScreen from "./src/app/ZoneDetailScreen";
import AlertsScreen    from "./src/app/AlertsScreen";
import ReportScreen    from "./src/app/ReportScreen";
import { colours, font } from "./src/theme";

const Tab   = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

const screenOptions = {
  headerStyle:       { backgroundColor: colours.bg1 },
  headerTintColor:   colours.textPrimary,
  headerTitleStyle:  { fontWeight: font.bold, fontSize: 15 },
  contentStyle:      { backgroundColor: colours.bg0 },
};

function ZonesStack() {
  return (
    <Stack.Navigator screenOptions={screenOptions}>
      <Stack.Screen
        name="Home"
        component={HomeScreen}
        options={{ title: "🏔 LAND-JEPA · Zones", headerShown: false }}
      />
      <Stack.Screen
        name="ZoneDetail"
        component={ZoneDetailScreen}
        options={({ route }) => ({
          title: route.params?.zone?.zone_id ?? "Zone Detail",
        })}
      />
    </Stack.Navigator>
  );
}

const TAB_ICONS = {
  Zones:  { active: "🏔", inactive: "🏔" },
  Alerts: { active: "🚨", inactive: "🚨" },
  Report: { active: "📝", inactive: "📝" },
};

export default function App() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({ route }) => ({
          ...screenOptions,
          headerShown: false,
          tabBarStyle: {
            backgroundColor: colours.bg1,
            borderTopColor:  colours.border,
            borderTopWidth:  1,
            paddingBottom:   6,
            height:          60,
          },
          tabBarActiveTintColor:   colours.accent,
          tabBarInactiveTintColor: colours.textMuted,
          tabBarLabelStyle: { fontSize: 11, fontWeight: font.medium },
          tabBarIcon: ({ focused }) => (
            <Text style={{ fontSize: 20 }}>
              {TAB_ICONS[route.name]?.active}
            </Text>
          ),
        })}
      >
        <Tab.Screen name="Zones"  component={ZonesStack} />
        <Tab.Screen name="Alerts" component={AlertsScreen} />
        <Tab.Screen
          name="Report"
          component={ReportScreen}
          options={{ title: "Field Report" }}
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
