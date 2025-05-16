<template>
  <transition
      name="page"
      @before-enter="beforeEnter"
      @enter="enter"
      @after-enter="afterEnter"
      @before-leave="beforeLeave"
      @leave="leave"
      @after-leave="afterLeave"
  >
    <slot></slot>
  </transition>
</template>

<script setup>
import {onMounted} from 'vue';

const beforeEnter = (el) => {
  el.style.opacity = 0;
  el.style.transform = 'translateY(20px)';
};

const enter = (el, done) => {
  const animation = el.animate([
    {opacity: 0, transform: 'translateY(20px)'},
    {opacity: 1, transform: 'translateY(0)'}
  ], {
    duration: 300,
    easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
    fill: 'forwards'
  });

  animation.onfinish = done;
};

const afterEnter = (el) => {
  el.style.opacity = '';
  el.style.transform = '';
};

const beforeLeave = (el) => {
  el.style.opacity = 1;
  el.style.transform = 'translateY(0)';
};

const leave = (el, done) => {
  const animation = el.animate([
    {opacity: 1, transform: 'translateY(0)'},
    {opacity: 0, transform: 'translateY(-20px)'}
  ], {
    duration: 300,
    easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
    fill: 'forwards'
  });

  animation.onfinish = done;
};

const afterLeave = (el) => {
  el.style.opacity = '';
  el.style.transform = '';
};

onMounted(() => {
  // 确保初始动画效果
  const mainContent = document.querySelector('.VPDoc');
  if (mainContent) {
    mainContent.style.animation = 'slideUp 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
  }
});
</script>

<style scoped>
@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style> 