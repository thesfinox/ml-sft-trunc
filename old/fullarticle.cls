\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{fullarticle}[2020/06/15 Article w/ Preloaded Packages]
\LoadClass{article}

%---- packages

\RequirePackage[utf8]{inputenc} %--------------- support for UTF-8 characters
\RequirePackage[T1]{fontenc} %------------------ main fonts for pdfLaTeX
\RequirePackage{geometry} %--------------------- manipulate page geometry
\RequirePackage{graphicx} %--------------------- pictures and figures
\RequirePackage{booktabs} %--------------------- nice tables
\RequirePackage{multicol} %--------------------- multi-column environment
\RequirePackage{multirow} %--------------------- multi-row environment
\RequirePackage{caption} %---------------------- better captions
\RequirePackage[usenames,dvipsnames]{xcolor} %-- colours
\RequirePackage{fancyhdr} %--------------------- fancy layout
\RequirePackage{authblk} %---------------------- manage multiple authors
\RequirePackage[nottoc]{tocbibind} %------------ put bibliography in TOC
\RequirePackage[backend=biber,
                sorting=none,
                style=ieee,
                maxnames=3]{biblatex} %--------- bibliography backend
\RequirePackage{bookmark} %--------------------- hyperref and links

%---- pass options to class

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

%---- end options

\ProcessOptions\relax

%---- numbers and style

\pagenumbering{arabic}
\pagestyle{fancy}

\setlength{\affilsep}{2em}
\renewcommand{\Authsep}{~,~}
\renewcommand{\Authand}{~~and~}
\renewcommand{\Authands}{~~and~}

%---- metadata

\makeatletter
\hypersetup
{%
  pdftitle={\@title},
  pdfstartview={FitH},
  pdfcreator={pdfLaTeX},
  pdfproducer={pdfLaTeX},
  pdflang={en-GB},
  pdfpagemode={UseOutlines},
  bookmarksopen={true},
  bookmarksnumbered={true},
  hidelinks,
  linktocpage,
  colorlinks={true},
  linkcolor={blue},
  urlcolor={blue},
  filecolor={blue},
  runcolor={blue},
  menucolor={blue},
  citecolor={red}
}
\makeatother

%---- title page

\providecommand{\email}[1]{\href{mailto:#1}{\nolinkurl{#1}}}

% vim: ft=tex
