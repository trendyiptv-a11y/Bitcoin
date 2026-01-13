import { Tabs } from 'expo-router';
import { BarChart3, TrendingUp, Settings } from 'react-native-svg';

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: true,
        headerStyle: {
          backgroundColor: '#1a1a1a',
        },
        headerTintColor: '#fff',
        headerTitleStyle: {
          fontWeight: '600',
        },
        tabBarStyle: {
          backgroundColor: '#1a1a1a',
          borderTopColor: '#333',
          borderTopWidth: 1,
        },
        tabBarActiveTintColor: '#00d4ff',
        tabBarInactiveTintColor: '#666',
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: 'Dashboard',
          tabBarLabel: 'Dashboard',
        }}
      />
      <Tabs.Screen
        name="indicators"
        options={{
          title: 'Indicatori',
          tabBarLabel: 'Indicatori',
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: 'Setări',
          tabBarLabel: 'Setări',
        }}
      />
    </Tabs>
  );
}
