(function($) {
  "use strict"; // Start of use strict

  // Smooth scrolling using jQuery easing
  $('a.js-scroll-trigger[href*="#"]:not([href="#"])').click(function() {
    if (location.pathname.replace(/^\//, '') == this.pathname.replace(/^\//, '') && location.hostname == this.hostname) {
      var target = $(this.hash);
      target = target.length ? target : $('[name=' + this.hash.slice(1) + ']');
      if (target.length) {
        $('html, body').animate({
          scrollTop: (target.offset().top - 48)
        }, 1000, "easeInOutExpo");
        return false;
      }
    }
  });

  // Closes responsive menu when a scroll trigger link is clicked
  $('.js-scroll-trigger').click(function() {
    $('.navbar-collapse').collapse('hide');
  });

  // Activate scrollspy to add active class to navbar items on scroll
  $('body').scrollspy({
    target: '#mainNav',
    offset: 54
  });

  // Collapse Navbar
  var navbarCollapse = function() {
    if ($("#mainNav").offset().top > 100) {
      $("#mainNav").addClass("navbar-shrink");
    } else {
      $("#mainNav").removeClass("navbar-shrink");
    }
  };
  // Collapse now if page is not at top
  navbarCollapse();
  // Collapse the navbar when page is scrolled
  $(window).scroll(navbarCollapse);

})(jQuery); // End of use strict


const contentFileInput = document.querySelector('input[name="content-file"]');
  const imageUrlInput = document.querySelector('input[name="image-url"]');

  imageUrlInput.addEventListener('input', () => {
    if (imageUrlInput.value) {
      contentFileInput.value = null;
    }
  });

  contentFileInput.addEventListener('input', () => {
    if (contentFileInput.value) {
      imageUrlInput.value = null;
    }
  });

  // Clear file input when image URL input is selected
  imageUrlInput.addEventListener('click', () => {
    contentFileInput.value = null;
  });

<script type="text/javascript">// <![CDATA[
        function loading() {
            $("#loading").show();
            $("#loading-gif").show();
            $("#content").hide();
        }
    // ]]></script>
    const form = document.querySelector('#caption-form');
    const imagePreview = document.querySelector('#img-preview');
    const caption = document.querySelector('#caption');
    const errorMessage = document.querySelector('#error-message');
    const successMessage = document.querySelector('#success-message');
                document.addEventListener('DOMContentLoaded', function() {
  previewImage();
});

    form.addEventListener('submit', (e) => {
      e.preventDefault();
      errorMessage.innerText = '';
      successMessage.innerText = '';
      caption.innerText = 'Generating caption...';

      const formData = new FormData();
      formData.append('image', imagePreview.src);

      fetch('/generate-caption', {
        method: 'POST',
        body: formData
      })
      .then(response => response.json())
      .then(data => {
        if (data.error) {
          caption.innerText = '';
          errorMessage.innerText = data.error;
        } else {
          caption.innerHTML = '<span>Generated Caption:</span> ' + data.caption;
          successMessage.innerText = 'Caption generated successfully!';
        }
      })
      .catch(error => {
        caption.innerText = '';
        errorMessage.innerText = 'An error occurred while generating caption. Please try again later.';
      });
    });
    
  function previewImage() {
  const fileInput = document.getElementById('content-file');
  const urlInput = document.getElementById('image-url');
  const previewImg = document.getElementById('img-preview');
  
  fileInput.addEventListener('change', function() {
    const file = fileInput.files[0];
    const reader = new FileReader();
    
    reader.addEventListener('load', function() {
      previewImg.src = reader.result;
    });
    
    reader.readAsDataURL(file);
  });
  
  urlInput.addEventListener('input', function() {
    previewImg.src = urlInput.value;
  });
}